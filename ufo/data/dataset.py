"""Long-sequence sampler over the cached Waymo scenes (UFO protocol).

A sequence spans ``seq_frames`` consecutive frames (2 s = 20, 8 s = 80,
16 s = 160 at 10 Hz).  Following UFO / STORM, every ``ctx_stride``-th frame
(0, 5, 10, ...) is a context frame fed to the recurrent update, one recurrent
step per context timestamp; all remaining frames are novel-view targets used
for supervision and evaluation.

Every pose is re-expressed relative to the ego pose of the sequence's first
frame so float32 keeps millimetre precision.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

FRAME_RATE = 10.0


class SceneCache:
    def __init__(self, path: str) -> None:
        self.path = path
        meta = np.load(os.path.join(path, "meta.npz"))
        self.frames = meta["frames"]
        self.intrinsics = meta["intrinsics"].astype(np.float32)
        self.cam_to_ego = meta["cam_to_ego"]
        self.ego_to_world = meta["ego_to_world"]
        boxes = np.load(os.path.join(path, "boxes.npz"))
        self.obj_to_world = boxes["obj_to_world"]
        self.size = boxes["size"]
        self.valid = boxes["valid"]
        self.lidar_count = boxes["lidar_count"]
        self.cls = boxes["cls"]
        self._arrays: dict[str, np.ndarray] = {}

    def array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(os.path.join(self.path, f"{name}.npy"), mmap_mode="r")
        return self._arrays[name]

    def __len__(self) -> int:
        return len(self.frames)


class UFOSequences(Dataset):
    """Args:
        cache_root:   output of :mod:`ufo.data.preprocess`.
        split_file:   PointForward split json (``train`` / ``val`` scene lists).
        split:        which list to use.
        seq_frames:   frames per sequence.
        ctx_stride:   context every n-th frame.
        num_target:   training only -- target frames sampled per sequence
                      (evaluation renders all of them).
        max_objects:  box budget per sequence (UFO: 32), ranked by LiDAR points.
        windows:      evaluation only -- explicit start frames; default
                      non-overlapping windows from frame 0.
        load_prior:   also return the precomputed depth prior (``prior.npy``,
                      :mod:`ufo.data.depth_prior`) of the *context* frames.
    """

    def __init__(
        self,
        cache_root: str,
        split_file: str,
        split: str = "train",
        seq_frames: int = 20,
        ctx_stride: int = 5,
        num_target: int = 4,
        max_objects: int = 32,
        train: bool = True,
        eval_span: int | None = None,
        seed: int = 0,
        load_prior: bool = False,
    ) -> None:
        with open(split_file) as fh:
            scenes = json.load(fh)[split]
        self.scenes = [
            SceneCache(os.path.join(cache_root, s))
            for s in scenes
            if os.path.exists(os.path.join(cache_root, s, "boxes.npz"))
        ]
        self.names = [os.path.basename(s.path) for s in self.scenes]
        self.seq_frames = seq_frames
        self.ctx_stride = ctx_stride
        self.num_target = num_target
        self.max_objects = max_objects
        self.train = train
        self.seed = seed
        self.load_prior = load_prior

        self.index: list[tuple[int, int]] = []
        for si, sc in enumerate(self.scenes):
            n = len(sc)
            if train:
                for start in range(0, n - seq_frames + 1):
                    self.index.append((si, start))
            else:
                # Non-overlapping windows over the first ``eval_span`` frames, so
                # 2 s / 8 s / 16 s all score the same 16 s of every val log.
                span = min(eval_span or n, n)
                for start in range(0, span - seq_frames + 1, seq_frames):
                    self.index.append((si, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        si, start = self.index[idx]
        sc = self.scenes[si]
        rng = np.random.default_rng(None if self.train else self.seed + idx)
        sl = np.arange(start, start + self.seq_frames)
        ctx = sl[:: self.ctx_stride]
        tgt_all = np.setdiff1d(sl, ctx)
        if self.train:
            tgt = np.sort(rng.choice(tgt_all, size=min(self.num_target, len(tgt_all)), replace=False))
        else:
            tgt = tgt_all

        world_from = np.linalg.inv(sc.ego_to_world[start])
        times = (sl - start) / FRAME_RATE

        def views(frames: np.ndarray, with_gt: bool) -> dict:
            c2w = world_from @ sc.ego_to_world[frames][:, None] @ sc.cam_to_ego[None]  # (T, C, 4, 4)
            out = {
                "images": torch.from_numpy(np.ascontiguousarray(sc.array("images")[frames])).permute(0, 1, 4, 2, 3).float() / 255.0,
                "c2w": torch.from_numpy(c2w).float(),
                "K": torch.from_numpy(np.broadcast_to(sc.intrinsics, (len(frames),) + sc.intrinsics.shape).copy()),
                "frame": torch.from_numpy(frames - start).long(),
                "time": torch.from_numpy((frames - start) / FRAME_RATE).float(),
            }
            if self.load_prior and not with_gt:
                out["prior"] = torch.from_numpy(np.asarray(sc.array("prior")[frames], dtype=np.float32))
            if with_gt:
                out["depth"] = torch.from_numpy(np.asarray(sc.array("depth")[frames], dtype=np.float32))
                out["sky"] = torch.from_numpy(np.asarray(sc.array("sky")[frames])).float()
                if not self.train:
                    out["dynamic"] = torch.from_numpy(np.asarray(sc.array("dynamic")[frames])).float()
            return out

        item = {
            "scene": self.names[si],
            "start": int(start),
            "context": views(ctx, with_gt=False),
            "target": views(tgt, with_gt=True),
            "seq_times": torch.from_numpy(times).float(),
        }
        item["boxes"] = self._boxes(sc, sl, world_from, rng)
        return item

    def _boxes(self, sc: SceneCache, sl: np.ndarray, world_from: np.ndarray, rng) -> dict:
        """Per-sequence object tracks: (O, F) poses in the sequence frame.

        Objects are ranked by total LiDAR points over the sequence (UFO keeps
        the 32 best-supported boxes).  Frames where a track is not annotated
        hold the nearest annotated pose so motion stays defined; ``valid``
        marks the real annotations.
        """
        o_max, f = self.max_objects, len(sl)
        pose = np.tile(np.eye(4, dtype=np.float32), (o_max, f, 1, 1))
        size = np.ones((o_max, f, 3), np.float32)
        valid = np.zeros((o_max, f), bool)
        present = np.zeros(o_max, bool)
        if len(sc.valid):
            support = (sc.lidar_count[:, sl] * sc.valid[:, sl]).sum(1)
            cand = np.nonzero((support > 0) & sc.valid[:, sl].any(1))[0]
            cand = cand[np.argsort(-support[cand])][:o_max]
            if self.train:
                # Slot identity is arbitrary; shuffling stops the model from
                # attaching meaning to slot order.
                slots = rng.permutation(o_max)[: len(cand)]
            else:
                slots = np.arange(len(cand))
            for slot, oi in zip(slots, cand):
                v = sc.valid[oi, sl]
                ids = np.nonzero(v)[0]
                nearest = ids[np.abs(np.arange(f)[:, None] - ids[None]).argmin(1)]
                pose[slot] = (world_from @ sc.obj_to_world[oi, sl][nearest]).astype(np.float32)
                size[slot] = sc.size[oi, sl][nearest]
                valid[slot] = v
                present[slot] = True
        return {
            "pose": torch.from_numpy(pose),
            "size": torch.from_numpy(size),
            "valid": torch.from_numpy(valid),
            "present": torch.from_numpy(present),
        }


def data_sources(d: dict) -> list[dict]:
    """The (name, cache_root, split_file) sources of a config's ``data`` block:
    ``data.sources`` when given (several datasets trained together), else the
    single legacy ``cache_root`` / ``split_file`` pair."""
    if d.get("sources"):
        return [dict(s) for s in d["sources"]]
    return [{"name": "waymo", "cache_root": d["cache_root"], "split_file": d["split_file"]}]


def build_sequences(cfg: dict, split: str, train: bool, **kwargs) -> list[tuple[str, UFOSequences]]:
    """One :class:`UFOSequences` per data source, named."""
    d = cfg["data"]
    load_prior = bool(cfg["model"].get("depth_prior", False))
    return [
        (s["name"], UFOSequences(s["cache_root"], s["split_file"], split, ctx_stride=d.get("ctx_stride", 5),
                                 max_objects=cfg["model"].get("max_objects", 32), train=train,
                                 load_prior=load_prior, **kwargs))
        for s in data_sources(d)
    ]


def build_train_set(cfg: dict) -> Dataset:
    """All sources' training sequences, concatenated."""
    d = cfg["data"]
    parts = [ds for _, ds in build_sequences(cfg, "train", True, seq_frames=d["seq_frames"],
                                            num_target=d["num_target"])]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def collate(batch: list[dict]) -> dict:
    """Stack fixed-shape fields; sequences in a batch share length and targets count."""

    def stack(d_list):
        return {k: torch.stack([d[k] for d in d_list]) for k in d_list[0]}

    return {
        "scene": [b["scene"] for b in batch],
        "start": [b["start"] for b in batch],
        "context": stack([b["context"] for b in batch]),
        "target": stack([b["target"] for b in batch]),
        "seq_times": torch.stack([b["seq_times"] for b in batch]),
        "boxes": stack([b["boxes"] for b in batch]),
    }
