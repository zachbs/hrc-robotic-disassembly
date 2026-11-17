"""Offline test of vp.select_detection() (gripper mask + nearest-to-target).

Fake YOLO results on a 640x480 frame, flat table 0.40 m below a camera
looking straight down. No RealSense, robot or YOLO needed.
  python test_detection_filter.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vgap_pipeline as vp  # noqa: E402

H, W = 480, 640
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
T_bc = np.eye(4)                       # camera at base (0,0,0.4), looking down
T_bc[:3, :3] = np.diag([1.0, -1.0, -1.0])
T_bc[:3, 3] = [0.0, 0.0, 0.40]
depth = np.full((H, W), 400, np.uint16)  # table everywhere, mm


class _Arr:
  """Stands in for a torch tensor: .cpu().numpy()."""
  def __init__(self, a):
    self.a = np.asarray(a)

  def cpu(self):
    return self

  def numpy(self):
    return self.a


class _NS:
  def __init__(self, n=0, **kw):
    self.n = n
    self.__dict__.update(kw)

  def __len__(self):
    return self.n


def make(dets):
  """dets: list of (class_name, conf, (x1, y1, x2, y2)); masks = filled boxes."""
  names = sorted({d[0] for d in dets})
  masks = []
  for _, _, (x1, y1, x2, y2) in dets:
    m = np.zeros((H, W), np.float32)
    m[y1:y2, x1:x2] = 1
    masks.append(_Arr(m))
  boxes = _NS(len(dets), cls=np.array([names.index(d[0]) for d in dets], float),
              conf=np.array([d[1] for d in dets]),
              xyxy=[_Arr(d[2]) for d in dets])
  return _NS(names=dict(enumerate(names)), boxes=boxes, masks=_NS(data=masks))


def px_to_base(u, v):
  z = 0.40
  p = np.array([(u - 320) / 600 * z, (v - 240) / 600 * z, z, 1.0])
  return (T_bc @ p)[:3]


gb = next(n for n in ["gearbox", "Gearbox"] if vp.object_key(n))
lb = next(c for c in ["L_block", "l_block", "L", "L-block"] if vp.object_key(c))
print(f"class names used: {gb}, {lb}")

ok = True


def check(label, cond):
  global ok
  ok &= bool(cond)
  print(f"  {'PASS' if cond else 'FAIL'}  {label}")


gripper = (gb, 0.73, (0, 0, 110, 90))         # the gripper corner "gearbox"
obj = (lb, 0.60, (300, 220, 380, 280))         # real L in the middle
occl = (gb, 0.80, (450, 300, 560, 400))        # occluder labeled gearbox
target = px_to_base(340, 250)                  # known object position

print("1) gripper ghost alone -> dropped, no detection")
idx, info = vp.select_detection(make([gripper]), (H, W), depth, K)
check("idx None", idx is None)
check("1 gripper drop", info["n_dropped_gripper"] == 1)

print("2) gripper + real object, no target -> real object")
idx, info = vp.select_detection(make([gripper, obj]), (H, W), depth, K)
check("picks L (idx 1)", idx == 1)

print("3) gripper mask off -> ghost wins on confidence (old behaviour)")
idx, _ = vp.select_detection(make([gripper, obj]), (H, W), depth, K, gripper_roi=None)
check("picks ghost (idx 0)", idx == 0)

print("4) occluder (higher conf) + real object, no target -> occluder (old behaviour)")
idx, _ = vp.select_detection(make([obj, occl]), (H, W), depth, K)
check("picks occluder (idx 1)", idx == 1)

print("5) same with target -> real object, occluder logged as not nearest")
idx, info = vp.select_detection(make([obj, occl]), (H, W), depth, K, T_bc, target)
check("picks L (idx 0)", idx == 0)
check("1 far drop", info["n_dropped_far"] == 1)
check(f"target dist small ({info['target_dist_mm']} mm)", info["target_dist_mm"] < 10)
print("     log:", info)

print("6) only the occluder, with target -> dropped as far (> 50 mm)")
idx, info = vp.select_detection(make([occl]), (H, W), depth, K, T_bc, target)
check("idx None", idx is None)
check("far drop logged", "far" in info["dropped"])

print("7) object shifted ~4 cm from target -> still kept")
idx, info = vp.select_detection(make([obj]), (H, W), depth, K, T_bc, px_to_base(400, 250))
check(f"kept ({info['target_dist_mm']} mm)", idx == 0)

print("8) --target-max-dist 0 (no limit) keeps far occluder")
idx, _ = vp.select_detection(make([occl]), (H, W), depth, K, T_bc, target, max_dist=0)
check("kept", idx == 0)

print("9) object partly in the corner (<50% inside) is kept")
part = (gb, 0.9, (60, 40, 260, 200))
idx, _ = vp.select_detection(make([part]), (H, W), depth, K)
check("kept", idx == 0)

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
