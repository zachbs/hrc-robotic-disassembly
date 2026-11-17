"""Compares pose methods on the same simulated captures as sim_stress_test.py.

Same scenes, seeds, depth model and masks (the trained YOLO model where it works,
simulated YOLO masks otherwise) as the stress test, for the gearbox, L and T
(the hemisphere has no corners or features, so the keypoint methods do not
apply to it). Each method gives an initial pose, then the same gate decides.

  pca_raw      PCA of the masked scan vs PCA of the CAD, centroid to centroid;
               of the 4 right-handed axis sign choices, the one with the best
               3D fit is kept. No ICP. (the previous group's approach)
  pca_icp      same 4 PCA candidates, each refined with the object's ICP,
               best ICP fitness kept
  ransac_icp   FPFH RANSAC + ICP, plain gate (no 90 deg hypotheses, no fallback)
  current      the full vgap_pipeline path: RANSAC + ICP + hypotheses +
               table yaw-sweep fallback
  solvepnp     2D keypoints (simulated detector) + CAD 3D keypoints ->
               cv2.solvePnPRansac -> ICP
  feature_rot  lip/feature constraint method: 2 features (gearbox: the upper lip of
               each end; L: the two arm ends; T: stem end + crossbar centre),
               their 3D points from a depth patch fix position and one axis;
               the CAD is rotated about that axis in 10 deg steps, scored with
               C, the best 3 get ICP, the highest C is kept

Simulated keypoint detector (stand-in for a YOLO-pose model): a keypoint is
reported if it is inside the image and not hidden (ray-cast), with 2 px
Gaussian noise; 5% of reported keypoints are outliers anywhere in the
mask's box. Labels are assumed consistent up to the object's symmetry (an
optimistic assumption for the gearbox's identical corners).

Every method's pose then goes through iou_check with the object's tau_c and
delta. Only `current` uses the hypotheses and the fallback.

Run from the project root:  python code/vgap/sim_method_compare.py [workers] [reps]
Writes code/vgap/sim_method_results/{trials.csv,summary.txt}
"""
import contextlib
import copy
import csv
import io
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vgap_pipeline as vp  # noqa: E402
import sim_stress_test as st  # noqa: E402

OUT = os.path.join(HERE, "sim_method_results")
OBJECTS = ["gearbox", "l_block", "t_block"]
METHODS = ["pca_raw", "pca_icp", "ransac_icp", "current", "solvepnp", "feature_rot"]
PX_NOISE, OUTLIER_P = 2.0, 0.05


# ------------------------------------------------------------ keypoints (CAD frame, m)
def _box(hx, hy, hz):
  return [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


GB_LIP_TOP = [(sx * 74.0, 0.0, sz * 9.0) for sx in (-1, 1) for sz in (-1, 1)]
KEYPOINTS_MM = {
    # 8 body corners (lips excluded), 4 lip top-face points (outboard of the
    # fastener hole, whose centre is at x~67), 4 lip tips
    "gearbox": _box(55.8, 45.0, 32.4) + GB_LIP_TOP
    + [(sx * 76.5, 0.0, sz * 4.5) for sx in (-1, 1) for sz in (-1, 1)],
    # every vertex of the prisms
    "l_block": [(50, -16.67, z) for z in (-13.33, 13.33)] + [(50, -50, z) for z in (-13.33, 13.33)]
    + [(-50, -50, z) for z in (-13.33, 13.33)] + [(-16.67, -16.67, z) for z in (-13.33, 13.33)]
    + [(-16.67, 50, z) for z in (-13.33, 13.33)] + [(-50, 50, z) for z in (-13.33, 13.33)],
    "t_block": [(x, y, z) for (x, y) in [(-16.67, 16.67), (-50, 16.67), (-50, 50), (16.67, -50),
                                         (-16.67, -50), (16.67, 16.67), (50, 50), (50, 16.67)]
                for z in (-13.33, 13.33)],
}
# Feature pairs for feature_rot: for each feature, candidate CAD points (one
# per face the object could rest on); the detector reports the upper one.
FEATURES_MM = {
    "gearbox": [[(74.0, 0, 9.0), (74.0, 0, -9.0)], [(-74.0, 0, 9.0), (-74.0, 0, -9.0)]],
    "l_block": [[(45.0, -33.33, 13.33), (45.0, -33.33, -13.33)],
                [(-33.33, 45.0, 13.33), (-33.33, 45.0, -13.33)]],
    "t_block": [[(0.0, -45.0, 13.33), (0.0, -45.0, -13.33)],
                [(0.0, 33.33, 13.33), (0.0, 33.33, -13.33)]],
}


def cad_frame_offset(mesh_mm):
  """The STL frame is the CAD frame used by the pipeline; keypoints above are
  given about the bounding-box centre, so shift them into the STL frame."""
  bb = mesh_mm.get_axis_aligned_bounding_box()
  return np.asarray(bb.get_center())


# ------------------------------------------------------------ scene helpers
def visibility_scene(mesh, T_ow, props, T_wc):
  scene = o3d.t.geometry.RaycastingScene()
  for m, T in [(mesh, T_wc @ T_ow)] + [(pm, T_wc) for _, pm in props]:
    mm = copy.deepcopy(m)
    mm.transform(T)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mm))
  table = o3d.geometry.TriangleMesh.create_box(1.2, 1.2, 0.02)
  table.translate([-0.6, -0.6, -0.02])
  table.transform(T_wc)
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(table))
  return scene


def visible(scene, pts_cam, tol=0.002):
  d = np.linalg.norm(pts_cam, axis=1)
  rays = np.hstack([np.zeros_like(pts_cam), pts_cam / d[:, None]]).astype(np.float32)
  t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
  return t >= d - tol


def project(pts_cam):
  uv = (st.K @ pts_cam.T).T
  return uv[:, :2] / uv[:, 2:3]


def detect_keypoints(kp_cad, T_cad_cam, scene, mask, rng):
  """Simulated keypoint detector. Returns (idx, uv) of reported keypoints."""
  pc = (T_cad_cam[:3, :3] @ kp_cad.T).T + T_cad_cam[:3, 3]
  uv = project(pc)
  ok = visible(scene, pc) & (uv[:, 0] >= 0) & (uv[:, 0] < st.W) & \
      (uv[:, 1] >= 0) & (uv[:, 1] < st.H)
  idx = np.nonzero(ok)[0]
  uv = uv[idx] + rng.normal(0, PX_NOISE, (len(idx), 2))
  x1, y1, x2, y2 = st.box_of(mask)
  out = rng.random(len(idx)) < OUTLIER_P
  uv[out] = np.c_[rng.uniform(x1, x2, out.sum()), rng.uniform(y1, y2, out.sum())]
  return idx, uv


def depth_point(depth, mask, uv, r=3):
  u, v = int(round(uv[0])), int(round(uv[1]))
  y0, y1, x0, x1 = max(0, v - r), min(st.H, v + r + 1), max(0, u - r), min(st.W, u + r + 1)
  patch = depth[y0:y1, x0:x1].astype(float)
  m = mask[y0:y1, x0:x1] & (patch > 0)
  if m.sum() < 3:
    return None
  z = np.median(patch[m]) / 1000.0
  return np.array([(uv[0] - st.K[0, 2]) * z / st.K[0, 0], (uv[1] - st.K[1, 2]) * z / st.K[1, 1], z])


# ------------------------------------------------------------ methods (return T_cad_to_cam or None)
def _pca(points):
  mu = points.mean(0)
  _, _, vt = np.linalg.svd(points - mu, full_matrices=False)
  return mu, vt.T


def pca_candidates(scan, cad):
  ps = np.asarray(scan.points)
  mu_s, A_s = _pca(ps)
  mu_c, A_c = _pca(cad.dense_points)
  cands = []
  for s1 in (1, -1):
    for s2 in (1, -1):
      S = np.diag([s1, s2, s1 * s2 * np.linalg.det(A_s) * np.linalg.det(A_c)])
      R = A_s @ S @ A_c.T
      T = np.eye(4)
      T[:3, :3] = R
      T[:3, 3] = mu_s - R @ mu_c
      cands.append(T)   # T_cad_to_cam
  return cands


def m_pca_raw(scan, cad, p):
  best, best_d = None, np.inf
  cad_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad.dense_points))
  for T in pca_candidates(scan, cad):
    c = copy.deepcopy(cad_pcd).transform(T)
    d = np.mean(np.asarray(scan.compute_point_cloud_distance(c)))
    if d < best_d:
      best, best_d = T, d
  return best


def m_pca_icp(scan, cad, p):
  best, best_key = None, None
  for T in pca_candidates(scan, cad):
    Ts, fit, rmse = vp.icp_refine(scan, cad, np.linalg.inv(T), p)
    key = (fit, -rmse)
    if best_key is None or key > best_key:
      best, best_key = np.linalg.inv(Ts), key
  return best


def m_ransac_icp(scan, cad, p):
  T = vp.ransac_register(scan, cad, p, vp.compute_fpfh(scan, p), vp.compute_fpfh(cad.pcd, p))
  T, _, _ = vp.icp_refine(scan, cad, T, p)
  return np.linalg.inv(T)


def m_solvepnp(scan, cad, p, kp_cad, idx, uv):
  if len(idx) < 6:
    return None
  ok, rvec, tvec, inl = cv2.solvePnPRansac(
      kp_cad[idx].astype(np.float64), uv.astype(np.float64), st.K, None,
      reprojectionError=6.0, iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
  if not ok or inl is None or len(inl) < 5:
    return None
  ok, rvec, tvec = cv2.solvePnP(kp_cad[idx][inl[:, 0]].astype(np.float64),
                                uv[inl[:, 0]].astype(np.float64), st.K, None,
                                rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
  T = np.eye(4)
  T[:3, :3] = cv2.Rodrigues(rvec)[0]
  T[:3, 3] = tvec[:, 0]
  Ts, _, _ = vp.icp_refine(scan, cad, np.linalg.inv(T), p)
  return np.linalg.inv(Ts)


def m_feature_rot(obs, scan, cad, p, q, pts3d, step=10, keep=3):
  """q: 2 CAD feature points, pts3d: their measured camera-frame points."""
  if any(x is None for x in pts3d):
    return None
  a_c, a_s = q[1] - q[0], pts3d[1] - pts3d[0]
  R0 = vp._rotation_a_to_b(a_c, a_s)
  axis = a_s / np.linalg.norm(a_s)
  mid_c, mid_s = q.mean(0), (pts3d[0] + pts3d[1]) / 2
  cands = []
  for th in range(0, 360, step):
    R = vp._rot(axis, th) @ R0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mid_s - R @ mid_c
    cands.append((vp._score_c(obs, cad, T, p), T))
  cands.sort(key=lambda c: c[0], reverse=True)
  best, best_C = None, -1
  for _, T in cands[:keep]:
    Ts, _, _ = vp.icp_refine(scan, cad, np.linalg.inv(T), p)
    T2 = np.linalg.inv(Ts)
    C = vp._score_c(obs, cad, T2, p)
    if C > best_C:
      best, best_C = T2, C
  return best


# ------------------------------------------------------------ worker
_W = {}


def _init():
  st._init()
  _W.update(st._W)
  kps, feats = {}, {}
  for k in OBJECTS:
    m = o3d.io.read_triangle_mesh(os.path.join(st.ROOT, st.MESHES[k]))
    off = cad_frame_offset(m) / 1000.0
    kps[k] = np.asarray(KEYPOINTS_MM[k]) / 1000.0 + off
    feats[k] = [np.asarray(f) / 1000.0 + off for f in FEATURES_MM[k]]
  _W.update(kps=kps, feats=feats)


def trial(job):
  key, color, scen, tilt, rep, seed = job
  # --- identical scene and mask to sim_stress_test.trial
  rng = np.random.default_rng(seed)
  o3d.utility.random.seed(int(seed % 2**31))
  mesh, cad = _W["meshes"][key], _W["cads"][key]
  T_ow, props, yaw, prop_tilt = st.place_object(mesh, scen, rng)
  center = np.asarray(copy.deepcopy(mesh).transform(T_ow).vertices).mean(0)
  target = center + np.r_[rng.uniform(-0.02, 0.02, 2), 0]
  T_wc = st.camera_pose(tilt, rng.uniform(0, 360), 0.40, target)
  bgr, depth, true_mask = st.render(mesh, T_ow, props, T_wc, color, rng)
  row = dict(object=key, color=color, scenario=scen, tilt=tilt, rep=rep,
             prop_tilt=round(prop_tilt, 1))
  if true_mask.sum() < 300:
    row["skip"] = True
    return row
  mask, box, src = None, None, "sim"
  res = _W["yolo"](bgr, conf=vp.CAMERA_SETTINGS["yolo_conf"], verbose=False)[0]
  idx = vp._best_known_detection(res)
  if idx is not None and res.masks is not None:
    if vp.object_key(res.names[int(res.boxes.cls[idx])]) == key:
      m = res.masks.data[idx].cpu().numpy()
      m = cv2.resize(m, (st.W, st.H), interpolation=cv2.INTER_NEAREST) > 0.5
      if vp.mask_iou(m, true_mask) > 0.5:
        mask, box, src = m, [int(v) for v in res.boxes.xyxy[idx].cpu().numpy()], "yolo"
  if mask is None:
    mask = st.sim_yolo_mask(true_mask, rng)
    box = st.box_of(mask)
  row["mask_source"] = src
  obs = vp.Observation(color=bgr, depth=depth, mask=mask, box=box, class_name=key,
                       object_key=key, confidence=1.0, K=st.K)
  T_gt = T_wc @ T_ow
  p = vp.OBJECT_CONFIGS[key]
  p_plain = dict(p, hypotheses=[], fallback=None)
  krng = np.random.default_rng(seed + 777)
  scene = visibility_scene(mesh, T_ow, props, T_wc)

  # keypoint detections
  kp_idx, kp_uv = detect_keypoints(_W["kps"][key], T_gt, scene, mask, krng)
  row["n_keypoints"] = len(kp_idx)
  # features: report the upper candidate (higher world z) of each feature
  q, pts3d = [], []
  for cand in _W["feats"][key]:
    zw = [(T_ow[:3, :3] @ c + T_ow[:3, 3])[2] for c in cand]
    c = cand[int(np.argmax(zw))]
    q.append(c)
    pc = T_gt[:3, :3] @ c + T_gt[:3, 3]
    if not visible(scene, pc[None])[0]:
      pts3d.append(None)
      continue
    uv = project(pc[None])[0] + krng.normal(0, PX_NOISE, 2)
    pts3d.append(depth_point(depth, mask, uv))
  row["features_ok"] = all(x is not None for x in pts3d)

  with contextlib.redirect_stdout(io.StringIO()):
    scan = vp.build_point_cloud(obs, p)
    for meth in METHODS:
      t0 = time.time()
      T, out = None, None
      if len(scan.points) < 50:
        pass
      elif meth == "current":
        out = vp.estimate_pose(obs, cad, p)
        row["current_fallback"] = bool(out.get("used_fallback"))
      else:
        if meth == "pca_raw":
          T = m_pca_raw(scan, cad, p)
        elif meth == "pca_icp":
          T = m_pca_icp(scan, cad, p)
        elif meth == "ransac_icp":
          T = m_ransac_icp(scan, cad, p)
        elif meth == "solvepnp":
          T = m_solvepnp(scan, cad, p, _W["kps"][key], kp_idx, kp_uv)
        elif meth == "feature_rot":
          T = m_feature_rot(obs, scan, cad, p, np.asarray(q), pts3d)
        if T is not None:
          out = vp.iou_check(obs, scan, cad, np.linalg.inv(T), p_plain)
      dt = time.time() - t0
      if out is None or "T_cad_to_cam" not in out:
        row.update({f"{meth}_state": "NoPose", f"{meth}_C": 0.0,
                    f"{meth}_rot": "", f"{meth}_tr": "", f"{meth}_s": round(dt, 2)})
        continue
      e_r, e_t = st.pose_error(T_gt, out["T_cad_to_cam"], cad)
      row.update({f"{meth}_state": out["state"], f"{meth}_C": round(out["C"], 3),
                  f"{meth}_rot": round(e_r, 2), f"{meth}_tr": round(e_t, 2),
                  f"{meth}_s": round(dt, 2)})
  return row


# ------------------------------------------------------------ summary
def _f(x):
  try:
    return float(x)
  except (TypeError, ValueError):
    return np.nan


def outcome(r, m):
  s = r[f"{m}_state"]
  er, et = _f(r[f"{m}_rot"]), _f(r[f"{m}_tr"])
  correct = er <= st.SUCCESS_ROT and et <= st.SUCCESS_TRANS
  bad = not (er <= st.BAD_ROT and et <= st.BAD_TRANS)
  if s == "NoPose":
    return "nopose"
  if s == "Valid":
    return "accept_ok" if correct else ("FALSE_ACCEPT" if bad else "accept_5to10")
  return "reject_bad" if bad else "reject_ok"


def summarize(rows):
  rows = [r for r in rows if not r.get("skip")]
  L = [f"Trials: {len(rows)} (gearbox, L, T x 3 colors x 4 scenes x 5 views x reps). "
       f"Correct = <= {st.SUCCESS_ROT} deg and <= {st.SUCCESS_TRANS} mm (symmetry-aware). "
       f"Bad = > {st.BAD_ROT} deg or > {st.BAD_TRANS} mm."]

  def table(title, subset):
    L.append(f"\n=== {title} (n={len(subset)}) ===")
    L.append(f"{'method':12s} {'raw ok%':>8s} {'accepted':>8s} {'acc ok%':>8s} {'FA':>4s} "
             f"{'rejBad':>6s} {'rejOK':>6s} {'noPose':>6s} {'medRot':>7s} {'medTr':>6s} {'sec':>5s}")
    for m in METHODS:
      oc = [outcome(r, m) for r in subset]
      raw = sum(_f(r[f"{m}_rot"]) <= st.SUCCESS_ROT and _f(r[f"{m}_tr"]) <= st.SUCCESS_TRANS
                for r in subset)
      acc = [r for r, o in zip(subset, oc) if o.startswith("accept") or o == "FALSE_ACCEPT"]
      er = [_f(r[f"{m}_rot"]) for r in acc]
      et = [_f(r[f"{m}_tr"]) for r in acc]
      n = max(len(subset), 1)
      L.append(f"{m:12s} {100*raw/n:8.1f} {len(acc):8d} {100*oc.count('accept_ok')/n:8.1f} "
               f"{oc.count('FALSE_ACCEPT'):4d} {oc.count('reject_bad'):6d} "
               f"{oc.count('reject_ok'):6d} {oc.count('nopose'):6d} "
               f"{np.nanmedian(er) if er else np.nan:7.2f} {np.nanmedian(et) if et else np.nan:6.2f} "
               f"{np.median([_f(r[f'{m}_s']) for r in subset]):5.1f}")

  table("ALL", rows)
  for k in OBJECTS:
    table(f"object={k}", [r for r in rows if r["object"] == k])
  for s in st.SCENARIOS:
    table(f"scene={s}", [r for r in rows if r["scenario"] == s])
  for t in st.TILTS:
    table(f"view={t}deg", [r for r in rows if int(r["tilt"]) == t])
  for c in st.COLORS:
    table(f"color={c}", [r for r in rows if r["color"] == c])
  L.append("\nraw ok% = pose correct before the gate; acc ok% = accepted by the gate AND "
           "correct (what the robot would act on); FA = accepted but bad; rejBad = bad "
           "pose correctly rejected; rejOK = fine pose rejected; noPose = method could not "
           "produce a pose (too few keypoints / feature hidden); sec = median runtime.")
  L.append(f"Keypoints reported per trial (median): "
           f"{np.median([_f(r['n_keypoints']) for r in rows]):.0f}; feature pair fully "
           f"visible in {sum(str(r['features_ok']) == 'True' for r in rows)}/{len(rows)} trials.")
  return "\n".join(L)


def main():
  workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
  reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
  os.makedirs(OUT, exist_ok=True)
  jobs, seed = [], 1000
  for k in st.MESHES:              # same seed sequence as sim_stress_test
    for c in st.COLORS:
      for s in st.SCENARIOS:
        for t in st.TILTS:
          for rep in range(reps):
            seed += 1
            if k in OBJECTS:
              jobs.append((k, c, s, t, rep, seed))
  print(f"{len(jobs)} trials on {workers} workers", flush=True)
  rows, t0 = [], time.time()
  with Pool(workers, initializer=_init) as pool:
    for i, r in enumerate(pool.imap_unordered(trial, jobs), 1):
      rows.append(r)
      if i % 20 == 0:
        print(f"  {i}/{len(jobs)} done, {time.time()-t0:.0f}s", flush=True)
  keys = sorted({k for r in rows for k in r})
  with open(os.path.join(OUT, "trials.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
  s = summarize([{k: str(v) for k, v in r.items()} for r in rows])
  with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write(s + "\n")
  print(s)


if __name__ == "__main__":
  main()
