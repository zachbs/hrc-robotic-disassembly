"""Ground-truth pose from a Gocator 3210 scan (independent of V-GAP).

The Gocator scan is registered to the CAD WITHOUT using V-GAP's pose, so a
wrong V-GAP pose cannot pull the reference toward itself:

  1. remove the table (RANSAC plane, keep points > 2 mm above it) and keep
     the largest cluster (drops the screw / stray points)
  2. coarse ICP from a grid of ~600 start orientations (50 "up" directions
     x 12 yaws, scan centroid on the CAD center), plus FPFH RANSAC starts
  3. fine point-to-plane ICP (1 mm) on the best candidates
  4. score = share of scan points within 1 mm of the CAD; the best is the
     reference pose. The best pose that is NOT the same pose (symmetry-aware,
     > 5 deg / 5 mm away) gives the margin: a big margin means the scan pins
     the pose down, a small one means the scan alone is ambiguous.

Registration runs scan -> CAD, as in vgap_pipeline (every scan point has a
CAD partner; the unseen side of the CAD does not matter). Units are meters.

  python gocator_gt.py --object l_block --scan gocator_test.ply   (offline check)
  python gocator_gt.py --sim                                       (synthetic test)
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vgap_pipeline as vp  # noqa: E402

reg = o3d.pipelines.registration

FINE_VOX = 0.001       # 1 mm (Gocator noise is ~0.01 mm, print tolerance ~0.2 mm)
COARSE_VOX = 0.003
FIT_DIST = 0.001       # "on the CAD" = within 1 mm


def _rot_from_up(up, yaw_deg):
  """Rotation taking CAD +z to `up`, then a yaw about `up`."""
  up = up / np.linalg.norm(up)
  z = np.array([0, 0, 1.0])
  v = np.cross(z, up)
  s, c = np.linalg.norm(v), z @ up
  if s < 1e-9:
    R0 = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
  else:
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R0 = np.eye(3) + s * K + (1 - c) * K @ K
  y = np.radians(yaw_deg)
  K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
  Ry = np.eye(3) + np.sin(y) * K + (1 - np.cos(y)) * K @ K
  return Ry @ R0


def _fib_dirs(n):
  i = np.arange(n) + 0.5
  phi = np.arccos(1 - 2 * i / n)
  th = np.pi * (1 + 5 ** 0.5) * i
  return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def table_in_sensor(T_base_sensor, table_z):
  """Table plane z = table_z (base frame) as (n, d) in the sensor frame."""
  R, t = T_base_sensor[:3, :3], T_base_sensor[:3, 3]
  n = R.T @ np.array([0.0, 0.0, 1.0])
  p = R.T @ (np.array([0.0, 0.0, table_z]) - t)
  return n, float(-n @ p)


def isolate_object(pts, plane_dist=0.0015, min_height=0.002, eps=0.004, near=None,
                   view_dir=(0.0, 0.0, 1.0), table=None):
  """Drop the table plane and everything below it; keep the largest cluster
  (or the one nearest `near`, sensor frame). view_dir = direction the sensor
  looks in its own frame (+z after gocator_cam's z flip; the gate eval passes
  the calibrated tool axis). table = (n, d) expected table plane in the sensor
  frame (from the arm pose + table height): used instead of searching for the
  table, refined on the table points near it when enough are seen. Needed when
  a big object (gearbox) fills the frame and little or no table is visible.
  Returns (points, plane or None)."""
  vd = np.asarray(view_dir, float)
  pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
  pcd = pcd.voxel_down_sample(0.0005)
  plane = None
  if table is not None:
    n, d = np.asarray(table[0], float), float(table[1])
    if n @ vd > 0:
      n, d = -n, -d
    P0 = np.asarray(pcd.points)
    near_t = np.nonzero(np.abs(P0 @ n + d) < 0.005)[0]
    if len(near_t) > 2000:
      m2, in2 = pcd.select_by_index(near_t).segment_plane(plane_dist, 3, 2000)
      n2, d2 = np.asarray(m2[:3]), m2[3]
      if n2 @ vd > 0:
        n2, d2 = -n2, -d2
      if len(in2) > 0.5 * len(near_t) and abs(n2 @ n) > 0.995:
        n, d = n2, d2
    h = P0 @ n + d
    pcd = pcd.select_by_index(np.nonzero(h > min_height)[0])
    plane = (n, d)
  elif len(pcd.points) > 500:
    model, inl = pcd.segment_plane(plane_dist, 3, 2000)
    if len(inl) > 0.15 * len(pcd.points):
      n, d = np.asarray(model[:3]), model[3]
      # The normal points toward the sensor when n . view_dir < 0. The table is the FARTHEST big flat
      # surface: a large object top (e.g. the T at 30-45 deg) can be the
      # biggest plane, with the real table beyond it. So while many points lie
      # beyond the current plane, fit a plane to them and move the table there.
      if n @ vd > 0:
        n, d = -n, -d
      P0 = np.asarray(pcd.points)
      for _ in range(3):
        beyond = np.nonzero(P0 @ n + d < -min_height)[0]
        if len(beyond) < max(2000, 0.05 * len(P0)):
          break
        far = pcd.select_by_index(beyond)
        m2, in2 = far.segment_plane(plane_dist, 3, 2000)
        n2, d2 = np.asarray(m2[:3]), m2[3]
        if n2 @ vd > 0:
          n2, d2 = -n2, -d2
        if len(in2) < 0.3 * len(beyond) or abs(n2 @ n) < 0.95:
          break
        n, d = n2, d2
      h = P0 @ n + d
      pcd = pcd.select_by_index(np.nonzero(h > min_height)[0])
      plane = (n, d)
  if len(pcd.points) < 50:
    raise RuntimeError(
      "Gocator: almost nothing above the largest flat surface. If that surface is "
      "the object's top, the table is out of range: the 3210 measures only "
      "~164-274 mm from its glass, so move in until table AND object are inside it.")
  lab = np.asarray(pcd.cluster_dbscan(eps, 20))
  if lab.max() < 0:
    return np.asarray(pcd.points), plane
  P = np.asarray(pcd.points)
  ids = [k for k in range(lab.max() + 1)]
  if near is not None:
    k = min(ids, key=lambda k: np.linalg.norm(P[lab == k].mean(0) - near))
  else:
    k = max(ids, key=lambda k: np.sum(lab == k))
  return P[lab == k], plane


class CadTarget:
  """CAD exterior points at 1 mm and 3 mm, with normals, for scan -> CAD ICP."""

  def __init__(self, cad, cfg=None):
    """cfg = the object's OBJECT_CONFIGS entry: for a mesh CAD the surface is
    resampled densely (the pipeline's 60k samples leave ~2.3 mm gaps on the big
    gearbox, so a perfect fit scored only ~0.78 'within 1 mm')."""
    self.cad = cad
    pts = cad.dense_points
    if cfg is not None and not str(cfg["cad_path"]).lower().endswith((".pcd", ".ply")):
      mesh = o3d.io.read_triangle_mesh(vp.resolve_path(cfg["cad_path"]))
      mesh.scale(0.001 if cfg["cad_units"] == "mm" else 1.0, center=np.zeros(3))
      pts = np.asarray(vp._exterior_samples(mesh, 400000).points)
    dense = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    self.fine = dense.voxel_down_sample(FINE_VOX)
    self.coarse = dense.voxel_down_sample(COARSE_VOX)
    for p, r in ((self.fine, 0.003), (self.coarse, 0.008)):
      p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(r, 30))
      n, x = np.asarray(p.normals), np.asarray(p.points)
      flip = np.sum(n * (x - cad.center), 1) < 0          # outward
      n[flip] *= -1
      p.normals = o3d.utility.Vector3dVector(n)
    self.tree = o3d.geometry.KDTreeFlann(self.fine)


def _icp(src, tgt, T, dist, iters, plane=True):
  est = (reg.TransformationEstimationPointToPlane() if plane
         else reg.TransformationEstimationPointToPoint())
  return reg.registration_icp(src, tgt, dist, T, est,
                              reg.ICPConvergenceCriteria(max_iteration=iters))


def _score(scan_fine, tgt, T):
  ev = reg.evaluate_registration(scan_fine, tgt.fine, FIT_DIST, T)
  return ev.fitness, ev.inlier_rmse


def gt_hemisphere(scan_pts, tgt, plane, radius=0.050, verbose=True):
  """Hemisphere: the dome is part of a sphere, so its own surface cannot fix
  the axis (any rotation about the sphere center fits). Center = known-radius
  sphere fit (exact); axis = table normal, which is only right when the
  hemisphere sits FLAT (axis_from_table). Propped: use the center only."""
  cadm = tgt.cad
  rng = np.random.default_rng(0)
  c, _ = vp._fit_sphere_known_radius(scan_pts, radius, 0.001, 3000, rng)
  n = plane[0] if plane is not None else -c / np.linalg.norm(c)
  R = vp._rotation_a_to_b(cadm.axial_axis, n)
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = c - R @ cadm.base_center
  s_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts)).voxel_down_sample(FINE_VOX)
  fit, rmse = _score(s_f, tgt, np.linalg.inv(T))
  out = dict(T_cad_to_sensor=T, fitness_1mm=float(fit), rmse_mm=float(rmse * 1000),
             n_points=len(s_f.points), second_fitness=float("nan"), margin=float("nan"),
             T_second=None, axis_from_table=plane is not None)
  if verbose:
    print(f"[+] Gocator GT (hemisphere): center from sphere fit, axis from "
          f"{'table normal' if plane is not None else 'view direction (no table!)'}; "
          f"within 1 mm {fit:.3f}, RMSE {rmse * 1000:.3f} mm")
  return out


def gt_pose(scan_pts, cad, plane=None, n_up=50, n_yaw=12, keep=12, seed=0, verbose=True):
  """Reference pose from an isolated object scan (sensor frame, m).
  plane: the table plane from isolate_object() (used for the hemisphere).

  Returns dict(T_cad_to_sensor, fitness_1mm, rmse_mm, margin, second_fitness,
  n_points, T_second)."""
  tgt = CadTarget(cad) if not isinstance(cad, CadTarget) else cad
  cadm = tgt.cad
  if cadm.axial_axis is not None:
    return gt_hemisphere(scan_pts, tgt, plane, verbose=verbose)
  scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts))
  s_f = scan.voxel_down_sample(FINE_VOX)
  s_c = scan.voxel_down_sample(COARSE_VOX)
  for p, r in ((s_f, 0.003), (s_c, 0.008)):
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(r, 30))
  cs = np.asarray(s_c.points).mean(0)

  starts = []
  for up in _fib_dirs(n_up):
    for yaw in np.arange(n_yaw) * 360.0 / n_yaw:
      R = _rot_from_up(up, yaw)          # sensor -> CAD rotation guess
      T = np.eye(4)
      T[:3, :3] = R
      T[:3, 3] = cadm.center - R @ cs
      starts.append(T)
  # FPFH RANSAC starts (cheap extra candidates)
  rng = np.random.default_rng(seed)
  try:
    feat = lambda p: reg.compute_fpfh_feature(  # noqa: E731
        p, o3d.geometry.KDTreeSearchParamHybrid(0.012, 100))
    fs, fc = feat(s_c), feat(tgt.coarse)
    for k in range(4):
      r = reg.registration_ransac_based_on_feature_matching(
          s_c, tgt.coarse, fs, fc, True, 0.006,
          reg.TransformationEstimationPointToPoint(False), 3,
          [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           reg.CorrespondenceCheckerBasedOnDistance(0.006)],
          reg.RANSACConvergenceCriteria(50000, 0.999), seed=int(rng.integers(1 << 30)))
      starts.append(r.transformation)
  except Exception:
    pass

  def run_coarse(starts):
    out = []
    for T in starts:
      r = _icp(s_c, tgt.coarse, T, 0.010, 15, plane=False)
      r = _icp(s_c, tgt.coarse, r.transformation, 0.005, 15)
      out.append((r.fitness, -r.inlier_rmse, r.transformation))
    return out

  coarse = run_coarse(starts)
  coarse.sort(key=lambda x: (x[0], x[1]), reverse=True)
  if plane is not None and coarse[0][0] < 0.95:
    # Fallback (11/02): when the part is cut off at the edge of the Gocator's
    # field of view (T at 30/45 deg), the scan centroid is far from the CAD
    # centroid and the centroid-aligned starts all miss. Use the table: try
    # the CAD's flat faces on the table normal (both ways up), every 10 deg of
    # yaw, with in-plane offsets up to +-30 mm.
    extra = []
    for up in (plane[0], -plane[0]):
      for yaw in np.arange(36) * 10.0:
        R = _rot_from_up(up, yaw)
        for dx in np.arange(-0.03, 0.031, 0.01):
          for dy in np.arange(-0.03, 0.031, 0.01):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cadm.center + np.array([dx, dy, 0.0]) - R @ cs
            extra.append(T)
    if verbose:
      print(f"[*] Gocator GT: best coarse fit {coarse[0][0]:.3f} (part cut off at the "
            f"field-of-view edge?); trying {len(extra)} table-based starts")
    coarse += run_coarse(extra)
    coarse.sort(key=lambda x: (x[0], x[1]), reverse=True)

  fine = []
  for _, _, T in coarse[:keep * 3]:
    if any(_same(T, F[2], cadm) for F in fine):
      continue
    r = _icp(s_f, tgt.fine, T, 0.003, 30)
    r = _icp(s_f, tgt.fine, r.transformation, 0.0015, 30)
    fit, rmse = _score(s_f, tgt, r.transformation)
    fine.append((fit, rmse, r.transformation))
    if len(fine) >= keep:
      break
  fine.sort(key=lambda x: (x[0], -x[1]), reverse=True)
  rests = True
  if plane is not None:
    # Physical check (11/01): from straight above the Gocator may see only the
    # top face, and the CAD then fits just as well with its BOTTOM face on it,
    # i.e. floating one thickness above the real part (a0 propped L: +26 mm).
    # Keep poses whose lowest CAD point touches the table: not through it
    # (> -3 mm) and not hovering (< 8 mm; propped parts rest one edge on it).
    n, d = plane
    P = np.asarray(tgt.fine.points)

    def low(T):
      Ts = np.linalg.inv(T)
      return float(((P @ Ts[:3, :3].T + Ts[:3, 3]) @ n + d).min())
    ok = [f for f in fine if -0.003 < low(f[2]) < 0.008]
    if ok and len(ok) < len(fine) and verbose:
      print(f"[*] Gocator GT: dropped {len(fine) - len(ok)} candidate pose(s) not resting on the table")
    if ok:
      fine = ok
    else:
      rests = False
      if verbose:
        print("[!] Gocator GT: no candidate rests on the table; check the scan")
  best = fine[0]
  second = next((f for f in fine[1:] if not _same(best[2], f[2], cadm)), None)
  out = dict(T_cad_to_sensor=np.linalg.inv(best[2]), fitness_1mm=float(best[0]),
             rmse_mm=float(best[1] * 1000), n_points=len(s_f.points),
             second_fitness=float(second[0]) if second else 0.0,
             margin=float(best[0] - (second[0] if second else 0.0)),
             T_second=np.linalg.inv(second[2]) if second else None, rests_on_table=rests)
  if verbose:
    print(f"[+] Gocator GT: {out['n_points']} pts, within 1 mm {out['fitness_1mm']:.3f}, "
          f"RMSE {out['rmse_mm']:.3f} mm, margin {out['margin']:.3f} "
          f"(next distinct pose {out['second_fitness']:.3f})")
  return out


def gt_refine(scan_pts, cad, T_cad_to_sensor):
  """Fine ICP from a known pose (repeat scans of the same view: how much
  does the reference move from scan to scan?). Returns (T, fitness, rmse_mm)."""
  tgt = CadTarget(cad) if not isinstance(cad, CadTarget) else cad
  if tgt.cad.axial_axis is not None:
    # hemisphere: only the sphere center is measured; keep the first axis
    c, _ = vp._fit_sphere_known_radius(scan_pts, 0.050, 0.001, 3000, np.random.default_rng(0))
    T = T_cad_to_sensor.copy()
    T[:3, 3] = c - T[:3, :3] @ tgt.cad.base_center
  else:
    s_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts)).voxel_down_sample(FINE_VOX)
    s_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.003, 30))
    r = _icp(s_f, tgt.fine, np.linalg.inv(T_cad_to_sensor), 0.003, 30)
    r = _icp(s_f, tgt.fine, r.transformation, 0.0015, 30)
    T = np.linalg.inv(r.transformation)
  s_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts)).voxel_down_sample(FINE_VOX)
  fit, rmse = _score(s_f, tgt, np.linalg.inv(T))
  return T, float(fit), float(rmse * 1000)


def _same(Ta, Tb, cad, rot=5.0, trans=5.0):
  """Same pose up to the CAD's symmetries. Ta, Tb: scan -> CAD."""
  import calibrate_tau_c_ur5e as cal
  e_r, e_t = cal.pose_error(np.linalg.inv(Ta), np.linalg.inv(Tb), cad)
  return e_r <= rot and e_t <= trans


# ==============================================================================
# synthetic test: does the GT search find the true pose from a partial scan?
# ==============================================================================
def _look_at(cam_pos, target):
  """T_world_sensor with sensor +z pointing from cam_pos to target."""
  z = target - cam_pos
  z /= np.linalg.norm(z)
  x = np.cross([0, 1.0, 0], z)
  if np.linalg.norm(x) < 1e-6:
    x = np.array([1.0, 0, 0])
  x /= np.linalg.norm(x)
  T = np.eye(4)
  T[:3, :3] = np.c_[x, np.cross(z, x), z]
  T[:3, 3] = cam_pos
  return T


def _sim_scan(cad, T_w_obj, T_w_s, noise=0.00003, rng=None):
  """Visible CAD points + table (world z = lowest object point), sensor frame."""
  rng = rng or np.random.default_rng(0)
  P = cad.dense_points @ T_w_obj[:3, :3].T + T_w_obj[:3, 3]
  T_s_w = np.linalg.inv(T_w_s)
  Ps = P @ T_s_w[:3, :3].T + T_s_w[:3, 3]
  pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ps))
  _, idx = pc.hidden_point_removal(np.zeros(3), 100.0)
  Ps = Ps[idx]
  zt = P[:, 2].min()
  g = np.stack(np.meshgrid(np.linspace(-0.10, 0.10, 160),
                           np.linspace(-0.08, 0.08, 130)), -1).reshape(-1, 2)
  c = T_w_obj[:3, 3]
  tab = np.c_[g + c[:2], np.full(len(g), zt)]
  from scipy.spatial import cKDTree
  d, _ = cKDTree(P[:, :2]).query(tab[:, :2])
  tab = tab[d > 0.003]                    # not under the object
  tab_s = tab @ T_s_w[:3, :3].T + T_s_w[:3, 3]
  out = np.vstack([Ps, tab_s])
  return out + rng.normal(0, noise, out.shape)


def sim_test(n=6, seed=1):
  import contextlib
  import io
  import calibrate_tau_c_ur5e as cal
  import sim_stress_test as st
  rng = np.random.default_rng(seed)
  print("object       placement  view  rot err  trans err  within1mm  margin")
  allok = True
  for k, rel in st.MESHES.items():
    params = dict(vp.OBJECT_CONFIGS[k])
    params["cad_path"], params["cad_units"] = os.path.join(st.ROOT, rel), "mm"
    with contextlib.redirect_stdout(io.StringIO()):
      cad = vp.load_cad_model(params)
    tgt = CadTarget(cad)
    for i in range(n):
      propped = i % 2 == 1
      R = vp._rot([0, 0, 1], rng.uniform(0, 360))
      if propped:
        R = vp._rot([1, 0, 0], rng.uniform(8, 20)) @ R
      T_w_obj = np.eye(4)
      T_w_obj[:3, :3] = R
      T_w_obj[:3, 3] = -R @ cad.center
      view = float(rng.choice([0, 30, 45]))
      az = rng.uniform(0, 2 * np.pi)
      d = 0.22
      cam = np.array([d * np.sin(np.radians(view)) * np.cos(az),
                      d * np.sin(np.radians(view)) * np.sin(az),
                      d * np.cos(np.radians(view))])
      T_w_s = _look_at(cam, np.zeros(3))
      T_true = np.linalg.inv(T_w_s) @ T_w_obj
      pts = _sim_scan(cad, T_w_obj, T_w_s, rng=rng)
      obj, plane = isolate_object(pts)
      out = gt_pose(obj, tgt, plane, verbose=False)
      e_r, e_t = cal.pose_error(T_true, out["T_cad_to_sensor"], cad)
      hemi_propped = cad.axial_axis is not None and propped
      ok = e_t < 1 and (e_r < 1 or hemi_propped)   # propped hemisphere: center only
      allok &= ok
      print(f"{k:12s} {'propped' if propped else 'flat':8s}  {view:4.0f}   "
            f"{e_r:6.2f}   {e_t:7.2f}    {out['fitness_1mm']:.3f}    {out['margin']:.3f}"
            f"  {'ok' if ok else 'FAIL'}")
  print("ALL OK" if allok else "SOME FAILED")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--object", choices=list(vp.OBJECT_CONFIGS))
  ap.add_argument("--scan", help="PLY in meters (gocator_cam.py output)")
  ap.add_argument("--sim", action="store_true")
  a = ap.parse_args()
  if a.sim:
    sim_test()
  else:
    cad = vp.load_cad_model(vp.OBJECT_CONFIGS[a.object])
    pts = np.asarray(o3d.io.read_point_cloud(a.scan).points)
    obj, plane = isolate_object(pts)
    print(f"[*] {len(pts)} points, {len(obj)} on the object, table "
          f"{'found' if plane else 'NOT found'}")
    out = gt_pose(obj, cad, plane)
    print(np.round(out["T_cad_to_sensor"], 5))
