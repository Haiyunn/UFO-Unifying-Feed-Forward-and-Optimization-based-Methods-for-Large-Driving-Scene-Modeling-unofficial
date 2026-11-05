"""CPU unit checks: quaternion round-trip (incl. ~180 deg) and motion blend == Eq. 6-7 for one-hot."""
import math

import torch

from ufo.geometry import mat_to_quat, quat_to_mat
from ufo.model import SceneMemory, UFO, UFOConfig

torch.manual_seed(0)
q = torch.nn.functional.normalize(torch.randn(10000, 4), dim=-1)
q[:100, 0] = 1e-4  # near-180 deg rotations
q = torch.nn.functional.normalize(q, dim=-1)
R = quat_to_mat(q)
err = (quat_to_mat(mat_to_quat(R)) - R).abs().max()
print(f"quat round-trip max err {err:.2e}")
assert err < 1e-3

model = UFO(UFOConfig(depth=1, dim=96, heads=4))
o, fseq, n = 3, 10, 50
yaw = torch.rand(o, fseq) * 2 * math.pi
pose = torch.eye(4).repeat(o, fseq, 1, 1)
pose[..., 0, 0], pose[..., 0, 1] = yaw.cos(), -yaw.sin()
pose[..., 1, 0], pose[..., 1, 1] = yaw.sin(), yaw.cos()
pose[..., :3, 3] = torch.randn(o, fseq, 3) * 20
boxes = {"pose": pose[None], "pose_inv": torch.linalg.inv(pose)[None]}
mem = SceneMemory(torch.randn(n, 3), torch.zeros(n, 96), torch.randint(0, fseq, (n,)), torch.zeros(n), torch.ones(n),
                  torch.zeros(n, 0, 3))
lab = torch.randint(0, o + 1, (n,))
A = torch.nn.functional.one_hot(lab, o + 1).float()
f = torch.tensor([2, 7])
T = model.motion(A, mem, boxes, 0, f)
# Literal Eq. 6-7 with one-hot assignment.
eye = torch.eye(4)
allp = torch.cat([eye.expand(1, fseq, 4, 4), pose])  # (1+O, F, 4, 4)
ref = torch.stack([allp[lab, fi] @ torch.linalg.inv(allp[lab, mem.f_ref]) for fi in f])
print(f"motion vs Eq.6-7 (one-hot) max err {(T - ref).abs().max():.2e}")
assert (T - ref).abs().max() < 1e-4
print("ok")
