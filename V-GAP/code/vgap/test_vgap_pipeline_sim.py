"""Offline test of vgap_pipeline.py on ray-cast (simulated) RealSense scans.

No camera or YOLO needed: each object is rendered on a table at 0.40 m with
the D435 intrinsics from IOU_pipeline.py, 0.7 mm depth noise, and the true
silhouette standing in for the YOLO mask. Reports symmetry-aware rotation
error, translation error, C and the gate decision for each run.

Run from the project root:  python code/vgap/test_vgap_pipeline_sim.py
"""

import copy
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vgap_pipeline as vp  # noqa: E402

K = np.array([[609.31, 0, 326.06], [0, 608.90, 249.21], [0, 0, 1.0]])
H, W = 480, 640
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MESHES = {
    "gearbox": "cad/gearbox/gearbox_assembled_153mm.stl",
    "l_block": "cad/l_block/l_block_100mm_reference.stl",
    "t_block": "cad/t_block/t_block_100mm_reference.stl",
    "hemisphere": "cad/hemisphere/hemisphere_100mm_reference.stl",
}


def camera_pose(tilt_deg, az_deg, dist):
  """World->camera for a camera on a sphere around the origin, looking at it."""
  t, a = np.radians(tilt_deg), np.radians(az_deg)
  cpos = dist * np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
  z = -cpos / np.linalg.norm(cpos)
  ref = np.array([0, 1.0, 0]) if tilt_deg < 1 else np.array([0, 0, 1.0])
  x = np.cross(z, ref) if tilt_deg >= 1 else np.array([1.0, 0, 0])
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  T_wc = np.eye(4)
  T_wc[:3, :3] = np.stack([x, y, z], 1)
  T_wc[:3, 3] = cpos
  return np.linalg.inv(T_wc)


def render(mesh_m, T_obj_world, T_world_cam, rng, noise=0.0007, drop=None):
  """Depth (uint16 mm) and object mask; the table is the plane z = 0."""
  obj = copy.deepcopy(mesh_m)
  obj.transform(T_world_cam @ T_obj_world)
  table = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 0.01)
  table.translate([-0.5, -0.5, -0.01])
  table.transform(T_world_cam)
  scene = o3d.t.geometry.RaycastingScene()
  oid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj))
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(table))
  rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
      o3d.core.Tensor(K), o3d.core.Tensor(np.eye(4)), W, H)
  ans = scene.cast_rays(rays)
  t = ans["t_hit"].numpy()
  hit = np.isfinite(t)
  z = np.where(hit, t * rays.numpy()[..., 5], 0)
  z = np.where(hit, z + rng.normal(0, noise, z.shape), 0)
  mask = hit & (ans["geometry_ids"].numpy() == oid)
  depth = (z * 1000).astype(np.uint16)
  if drop is not None:
    depth[drop(ans, mask)] = 0
  ys, xs = np.nonzero(mask)
  return depth, mask, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def run(key, tilts=(0, 30, 45), yaws=(0, 35), trials=2, seed=0):
  params = dict(vp.OBJECT_CONFIGS[key])
  params["cad_path"] = os.path.join(ROOT, MESHES[key])
  params["cad_units"] = "mm"
  cad = vp.load_cad_model(params)
  mesh = o3d.io.read_triangle_mesh(params["cad_path"])
  mesh.scale(0.001, center=np.zeros(3))
  # put the CAD frame so the object rests on z = 0, centered at the origin
  bb = mesh.get_axis_aligned_bounding_box()
  rest = np.eye(4)
  rest[:3, 3] = -np.array([*bb.get_center()[:2], bb.min_bound[2]])
  rng = np.random.default_rng(seed)
  o3d.utility.random.seed(seed)
  rows = []
  for tilt in tilts:
    for yaw in yaws:
      T_ow = np.eye(4)
      T_ow[:3, :3] = vp._rot([0, 0, 1], yaw)
      T_ow = T_ow @ rest
      T_wc = camera_pose(tilt, 20, 0.40)
      depth, mask, box = render(mesh, T_ow, T_wc, rng)
      obs = vp.Observation(color=np.zeros((H, W, 3), np.uint8), depth=depth,
                           mask=mask, box=box, class_name=key, object_key=key,
                           confidence=1.0, K=K)
      T_gt = T_wc @ T_ow
      for trial in range(trials):
        out = vp.estimate_pose(obs, cad, params)
        if "T_cad_to_cam" not in out:
          rows.append((key, tilt, yaw, trial, np.nan, np.nan, 0, out["state"]))
          continue
        T = out["T_cad_to_cam"]
        e_r = vp.symmetric_rotation_error_deg(T_gt[:3, :3], T[:3, :3], cad)
        if cad.axial_axis is not None:
          p_gt = T_gt[:3, :3] @ cad.base_center + T_gt[:3, 3]
          p_es = T[:3, :3] @ cad.base_center + T[:3, 3]
          e_t = np.linalg.norm(p_gt - p_es) * 1000
        else:
          e_t = min(np.linalg.norm((T_gt @ S)[:3, 3] - T[:3, 3])
                    for S in cad.symmetries) * 1000
        rows.append((key, tilt, yaw, trial, round(e_r, 1), round(e_t, 1),
                     round(out["C"], 3), f"{out['state']} {out['reason']}".strip()
                     + (" (fallback)" if out.get("used_fallback") else "")))
  return rows


if __name__ == "__main__":
  keys = sys.argv[1:] or list(MESHES)
  allrows = []
  for k in keys:
    allrows += run(k)
  print("\nobject     tilt yaw trial  rotErr(deg) transErr(mm)  C      gate")
  for r in allrows:
    print(f"{r[0]:10s} {r[1]:4d} {r[2]:3d} {r[3]:5d}  {r[4]:10} {r[5]:11}  "
          f"{r[6]:<6} {r[7]}")
