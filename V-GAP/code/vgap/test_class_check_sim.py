"""Offline test of the L/T class check on ray-cast scans.

Renders the L and T blocks at several views, then hands each scan to
estimate_pose_with_class_check() with YOLO's label set to the correct class
and to the wrong one. The check should pick the true object both times.

Run from the project root:  python code/vgap/test_class_check_sim.py
"""
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vgap_pipeline as vp  # noqa: E402
from test_vgap_pipeline_sim import K, H, W, ROOT, MESHES, camera_pose, render  # noqa: E402

vp.DEBUG_VIEWS = False
for k in ("l_block", "t_block"):
  vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(ROOT, MESHES[k])

rng = np.random.default_rng(0)
o3d.utility.random.seed(0)
cad_cache = {}
rows = []
for true in ("l_block", "t_block"):
  mesh = o3d.io.read_triangle_mesh(os.path.join(ROOT, MESHES[true]))
  mesh.scale(0.001, center=np.zeros(3))
  bb = mesh.get_axis_aligned_bounding_box()
  rest = np.eye(4)
  rest[:3, 3] = -np.array([*bb.get_center()[:2], bb.min_bound[2]])
  for tilt in (0, 30, 45, 60):
    for yaw in (0, 35, 120):
      T_ow = np.eye(4)
      T_ow[:3, :3] = vp._rot([0, 0, 1], yaw)
      T_ow = T_ow @ rest
      depth, mask, box = render(mesh, T_ow, camera_pose(tilt, 20, 0.40), rng)
      for label in ("l_block", "t_block"):
        obs = vp.Observation(color=np.zeros((H, W, 3), np.uint8), depth=depth,
                             mask=mask, box=box, class_name=label,
                             object_key=label, confidence=1.0, K=K)
        out = vp.estimate_pose_with_class_check(obs, cad_cache)
        rows.append((true, label, tilt, yaw, out["final_object"],
                     out["class_scores"], round(out.get("class_margin", 0), 3),
                     f"{out['state']} {out['reason']}".strip()))

print("\ntrue     yolo_label tilt yaw  chosen    correct  C per CAD                          margin  gate")
for t, l, tilt, yaw, ch, sc, m, g in rows:
  print(f"{t:8s} {l:10s} {tilt:4d} {yaw:3d}  {ch:8s}  {str(ch == t):7s}  {str(sc):34s} {m:6.3f}  {g}")
ok = sum(r[4] == r[0] for r in rows)
wrong = [r for r in rows if r[1] != r[0]]
print(f"\nCorrect class chosen: {ok}/{len(rows)}; "
      f"YOLO-wrong cases fixed: {sum(r[4] == r[0] for r in wrong)}/{len(wrong)}")
