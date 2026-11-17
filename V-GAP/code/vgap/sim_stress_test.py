"""Stress test of vgap_pipeline.py on realistic simulated RealSense captures.

No hardware needed. For every trial the scene (object, table, props,
clutter) is ray-cast from a wrist camera and turned into what the real
capture_observation() would produce:

  * RGB: Lambertian + specular shading per filament (white / grey / dark
    reflective), textured table, sensor noise. Fed to your YOLO model; if it
    misses or picks the wrong class (expected: it was trained on lab images,
    not renders), a simulated YOLO mask is used instead and the trial is
    marked mask_source = "sim".
  * Depth: 25-frame burst. Each frame has z^2-scaled noise, low-frequency
    noise, 1 mm quantisation, invalid pixels at depth edges, flying pixels,
    grazing-angle dropout, and colour-dependent dropout (the dark reflective
    print also loses near-normal pixels to IR saturation). Then the pipeline's
    own temporal_median_ignore_zeros + bilateral filter, as in capture.

Scenarios: flat on the table, propped on a screw (6 mm), propped on a block
(20 mm), and flat with a clutter box partly occluding it.
Views: 0/15/30/45/60 deg tilt from the top, random azimuth, 0.40 m.

Each trial is run with the main path only (RANSAC+ICP+hypotheses+gate, no
table fallback) and, if that is a Conflict, with the table yaw-sweep
fallback too, so both can be reported. L/T use the class check.

Run from the project root:  python code/vgap/sim_stress_test.py [n_workers]
Writes code/vgap/sim_stress_results/{trials.csv,summary.txt}
"""
import copy
import csv
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

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "sim_stress_results")
YOLO_PATH = os.path.join(ROOT, "10-08-26seg26s.pt")
K = np.array([[609.31, 0, 326.06], [0, 608.90, 249.21], [0, 0, 1.0]])
H, W = 480, 640
FRAMES = 25
MESHES = {
    "gearbox": "cad/gearbox/gearbox_assembled_153mm.stl",
    "l_block": "cad/l_block/l_block_100mm_reference.stl",
    "t_block": "cad/t_block/t_block_100mm_reference.stl",
    "hemisphere": "cad/hemisphere/hemisphere_100mm_reference.stl",
}
# albedo (BGR 0-1), specular strength, depth behaviour
COLORS = {
    "white": dict(rgb=(0.90, 0.91, 0.92), spec=0.05, drop=0.005, noise=1.0,
                  graze_deg=68, sat_deg=None),
    "grey": dict(rgb=(0.50, 0.50, 0.50), spec=0.10, drop=0.02, noise=1.2,
                 graze_deg=65, sat_deg=None),
    "dark_reflective": dict(rgb=(0.17, 0.15, 0.15), spec=0.7, drop=0.12,
                            noise=1.6, graze_deg=55, sat_deg=12),
}
SCENARIOS = ["flat", "screw", "block", "clutter"]
TILTS = [0, 15, 30, 45, 60]
SUCCESS_ROT, SUCCESS_TRANS = 5.0, 5.0      # deg, mm
BAD_ROT, BAD_TRANS = 10.0, 10.0            # worse than this = bad pose


# ------------------------------------------------------------------ scene
def camera_pose(tilt_deg, az_deg, dist, target):
  t, a = np.radians(tilt_deg), np.radians(az_deg)
  cpos = target + dist * np.array([np.sin(t) * np.cos(a),
                                   np.sin(t) * np.sin(a), np.cos(t)])
  z = (target - cpos) / np.linalg.norm(target - cpos)
  ref = np.array([np.cos(a + np.pi / 2), np.sin(a + np.pi / 2), 0.0])
  x = np.cross(z, ref) if tilt_deg >= 1 else ref
  x = x - (x @ z) * z
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  T_wc = np.eye(4)
  T_wc[:3, :3] = np.stack([x, y, z], 1)
  T_wc[:3, 3] = cpos
  return np.linalg.inv(T_wc)


def rest_transform(mesh):
  bb = mesh.get_axis_aligned_bounding_box()
  T = np.eye(4)
  T[:3, 3] = -np.array([*bb.get_center()[:2], bb.min_bound[2]])
  return T


def place_object(mesh, scenario, rng):
  """Returns T_obj_world and a list of prop meshes (world frame)."""
  T = vp._rot_about(np.zeros(3), [0, 0, 1], 0)
  T = rest_transform(mesh)
  yaw = rng.uniform(0, 360)
  Rz = np.eye(4)
  Rz[:3, :3] = vp._rot([0, 0, 1], yaw)
  T = Rz @ T
  props = []
  if scenario in ("screw", "block"):
    hp = 0.006 if scenario == "screw" else 0.020
    u_ang = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(u_ang), np.sin(u_ang), 0.0])      # raised side
    a = np.cross([0, 0, 1.0], u)                            # tilt axis
    v = np.asarray(copy.deepcopy(mesh).transform(T).vertices)
    span = (v @ u).max() - (v @ u).min()
    lever = 0.7 * span
    alpha = np.degrees(np.arcsin(min(hp / lever, 0.9)))
    R = np.eye(4)
    R[:3, :3] = vp._rot(a, -alpha)   # lifts the +u side
    T = R @ T
    v = np.asarray(copy.deepcopy(mesh).transform(T).vertices)
    T[2, 3] -= v[:, 2].min()
    v = np.asarray(copy.deepcopy(mesh).transform(T).vertices)
    low = v[np.argmin(v @ u)]
    p = low + lever * u
    if scenario == "screw":
      pr = o3d.geometry.TriangleMesh.create_cylinder(0.003, 0.045)
      pr.rotate(vp._rot(np.cross([0, 0, 1.0], a), 90), center=np.zeros(3))
      pr.translate([p[0], p[1], 0.003])
    else:
      pr = o3d.geometry.TriangleMesh.create_box(0.03, 0.03, hp)
      pr.translate([p[0] - 0.015, p[1] - 0.015, 0])
    props.append(("prop", pr))
    tilt = alpha
  else:
    tilt = 0.0
  if scenario == "clutter":
    v = np.asarray(copy.deepcopy(mesh).transform(T).vertices)
    c = v.mean(axis=0)
    r = 0.5 * np.linalg.norm(v[:, :2].max(0) - v[:, :2].min(0))
    ang = rng.uniform(0, 2 * np.pi)
    box = o3d.geometry.TriangleMesh.create_box(0.04, 0.04, 0.07)
    box.translate([c[0] + (r + 0.005) * np.cos(ang) - 0.02,
                   c[1] + (r + 0.005) * np.sin(ang) - 0.02, 0])
    props.append(("clutter", box))
  return T, props, yaw, tilt


def table_texture(rng):
  base = np.full((H, W), 0.55, np.float32)
  noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6)
  grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1)
  t = base + 0.06 * noise / (noise.std() + 1e-6) + 0.02 * grain
  return np.stack([t * 0.85, t * 0.95, t * 1.05], -1)


def render(obj_mesh, T_ow, props, T_wc, color, rng):
  scene = o3d.t.geometry.RaycastingScene()
  obj = copy.deepcopy(obj_mesh)
  obj.transform(T_wc @ T_ow)
  obj.compute_triangle_normals()
  oid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj))
  table = o3d.geometry.TriangleMesh.create_box(1.2, 1.2, 0.02)
  table.translate([-0.6, -0.6, -0.02])
  table.transform(T_wc)
  tid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(table))
  pids = []
  for _, pm in props:
    pm = copy.deepcopy(pm)
    pm.transform(T_wc)
    pids.append(scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(pm)))
  rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
      o3d.core.Tensor(K), o3d.core.Tensor(np.eye(4)), W, H)
  ans = scene.cast_rays(rays)
  r = rays.numpy()
  dirs = r[..., 3:] / np.linalg.norm(r[..., 3:], axis=-1, keepdims=True)
  t = ans["t_hit"].numpy()
  hit = np.isfinite(t)
  gid = ans["geometry_ids"].numpy()
  nrm = ans["primitive_normals"].numpy()
  z = np.where(hit, t * r[..., 5], 0.0)
  mask = hit & (gid == oid)
  cosi = np.abs(np.sum(nrm * dirs, -1))
  # ---- RGB
  light = -dirs * 0.6 + np.array([0.2, -0.5, 0.0])     # headlamp + room light
  light /= np.linalg.norm(light, axis=-1, keepdims=True)
  ndl = np.clip(np.abs(np.sum(nrm * light, -1)), 0, 1)
  rgb = np.zeros((H, W, 3), np.float32)
  tex = table_texture(rng)
  shade = 0.35 + 0.65 * ndl
  rgb[gid == tid] = (tex * shade[..., None])[gid == tid]
  for pid in pids:
    rgb[gid == pid] = (0.45 * shade[..., None] * np.ones(3))[gid == pid]
  c = COLORS[color]
  hvec = light - dirs
  hvec /= np.linalg.norm(hvec, axis=-1, keepdims=True)
  spec = c["spec"] * np.clip(np.abs(np.sum(nrm * hvec, -1)), 0, 1) ** 40
  obj_rgb = np.array(c["rgb"])[None, None] * shade[..., None] + spec[..., None]
  rgb[mask] = obj_rgb[mask]
  rgb = cv2.GaussianBlur(rgb, (3, 3), 0.6) + rng.normal(0, 0.012, rgb.shape)
  bgr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
  # ---- depth burst
  incid = np.degrees(np.arccos(np.clip(cosi, 0, 1)))
  sigma = 0.0008 * (z / 0.4) ** 2
  sigma = np.where(mask, sigma * c["noise"], sigma)
  gx = np.abs(cv2.Sobel(z.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
  gy = np.abs(cv2.Sobel(z.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
  edge = cv2.dilate(((gx + gy) > 0.03).astype(np.uint8), np.ones((3, 3))) > 0
  zmax = cv2.dilate(z.astype(np.float32), np.ones((5, 5)))
  graze = np.clip((incid - (c["graze_deg"] if True else 0)) / 20, 0, 1)
  p_drop = np.where(mask, c["drop"] + 0.85 * graze, 0.003 + 0.6 * np.clip((incid - 75) / 15, 0, 1))
  if c["sat_deg"] is not None:
    p_drop = np.where(mask & (incid < c["sat_deg"]), np.maximum(p_drop, 0.55), p_drop)
  burst = []
  for _ in range(FRAMES):
    low = cv2.resize(rng.normal(0, 0.0003, (H // 16, W // 16)).astype(np.float32),
                     (W, H), interpolation=cv2.INTER_CUBIC)
    zf = z + rng.normal(0, 1, z.shape) * sigma + low
    fly = edge & (rng.random(z.shape) < 0.15)
    mix = rng.random(z.shape)
    zf = np.where(fly, mix * z + (1 - mix) * zmax, zf)
    invalid = (~hit) | (rng.random(z.shape) < p_drop) | \
              (edge & (rng.random(z.shape) < 0.55))
    zf[invalid] = 0
    burst.append(np.round(zf * 1000).clip(0, 65535).astype(np.uint16))
  median = vp.temporal_median_ignore_zeros(burst)
  filt = cv2.bilateralFilter(median.astype(np.float32), d=5, sigmaColor=15.0,
                             sigmaSpace=5.0)
  filt[median == 0] = 0
  return bgr, filt.astype(np.uint16), mask


def sim_yolo_mask(mask, rng):
  """YOLO-like mask: 1/4-resolution prototype, upsampled, +-1-2 px boundary."""
  small = cv2.resize(mask.astype(np.float32), (W // 4, H // 4),
                     interpolation=cv2.INTER_AREA)
  m = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
  d = int(rng.integers(-1, 3))
  if d > 0:
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * d + 1,) * 2)) > 0
  elif d < 0:
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3))) > 0
  return m


def box_of(m):
  ys, xs = np.nonzero(m)
  return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ------------------------------------------------------------------ pipeline
def run_main_and_fallback(obs, cad, params):
  """estimate_pose() split so the no-fallback and fallback results are both kept."""
  p = dict(params)
  scan = vp.build_point_cloud(obs, p)
  if len(scan.points) < 50:
    r = dict(state="Conflict", reason="too few depth points", C=0.0)
    return r, r
  if p["global_method"] == "fpfh_ransac":
    T = vp.ransac_register(scan, cad, p, vp.compute_fpfh(scan, p),
                           vp.compute_fpfh(cad.pcd, p))
  else:
    T = vp.ransac_register(scan, cad, p, obs=obs)
  if p["icp_thresholds"]:
    T, _, _ = vp.icp_refine(scan, cad, T, p)
  main = vp.iou_check(obs, scan, cad, T, p)
  main["used_fallback"] = False
  fb = main
  if main["state"] == "Conflict" and p.get("fallback") == "table_yaw_sweep":
    T_fb = vp.table_yaw_sweep(obs, scan, cad, p)
    if T_fb is not None:
      out_fb = vp.iou_check(obs, scan, cad, T_fb, p)
      if out_fb["C"] > main["C"]:
        fb = dict(out_fb, used_fallback=True)
  return main, fb


def class_checked(obs, cads):
  keys = [obs.object_key] + vp.CLASS_RIVALS.get(obs.object_key, [])
  res = {k: run_main_and_fallback(obs, cads[k], vp.OBJECT_CONFIGS[k]) for k in keys}
  outs = []
  for i in (0, 1):  # 0 = main path, 1 = with fallback
    best = max(keys, key=lambda k: res[k][i].get("C", 0.0))
    o = dict(res[best][i])
    o["final_object"] = best
    if len(keys) > 1:
      ru = max(res[k][i].get("C", 0.0) for k in keys if k != best)
      if o["state"] == "Valid" and o["C"] - ru < vp.CLASS_DELTA:
        o["state"], o["reason"] = "Conflict", "ambiguous class"
    outs.append(o)
  return outs


def pose_error(T_gt, T, cad):
  if T is None:
    return np.nan, np.nan
  e_r = vp.symmetric_rotation_error_deg(T_gt[:3, :3], T[:3, :3], cad)
  if cad.axial_axis is not None:
    pg = T_gt[:3, :3] @ cad.base_center + T_gt[:3, 3]
    pe = T[:3, :3] @ cad.base_center + T[:3, 3]
    e_t = np.linalg.norm(pg - pe) * 1000
  else:
    e_t = min(np.linalg.norm((T_gt @ S)[:3, 3] - T[:3, 3])
              for S in cad.symmetries) * 1000
  return e_r, e_t


# ------------------------------------------------------------------ worker
_W = {}


def _init():
  import torch
  torch.set_num_threads(1)
  o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
  vp.DEBUG_VIEWS = False
  cads, meshes = {}, {}
  for k, rel in MESHES.items():
    vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(ROOT, rel)
    vp.OBJECT_CONFIGS[k]["cad_units"] = "mm"
    cads[k] = vp.load_cad_model(vp.OBJECT_CONFIGS[k])
    m = o3d.io.read_triangle_mesh(os.path.join(ROOT, rel))
    m.scale(0.001, center=np.zeros(3))
    meshes[k] = m
  yolo = None
  if os.path.isfile(YOLO_PATH):
    from ultralytics import YOLO
    yolo = YOLO(YOLO_PATH)
  _W.update(cads=cads, meshes=meshes, yolo=yolo)


def trial(job):
  import contextlib
  import io
  key, color, scen, tilt, rep, seed = job
  rng = np.random.default_rng(seed)
  o3d.utility.random.seed(int(seed % 2**31))
  mesh, cads = _W["meshes"][key], _W["cads"]
  T_ow, props, yaw, prop_tilt = place_object(mesh, scen, rng)
  center = np.asarray(copy.deepcopy(mesh).transform(T_ow).vertices).mean(0)
  target = center + np.r_[rng.uniform(-0.02, 0.02, 2), 0]
  T_wc = camera_pose(tilt, rng.uniform(0, 360), 0.40, target)
  bgr, depth, true_mask = render(mesh, T_ow, props, T_wc, color, rng)
  row = dict(object=key, color=color, scenario=scen, tilt=tilt, rep=rep,
             yaw=round(yaw, 1), prop_tilt=round(prop_tilt, 1),
             visible_px=int(true_mask.sum()))
  if true_mask.sum() < 300:
    row.update(mask_source="none", state_main="skip")
    return row
  # YOLO, same handling as capture_observation()
  mask, box, src, yolo_cls = None, None, "sim", ""
  if _W["yolo"] is not None:
    res = _W["yolo"](bgr, conf=vp.CAMERA_SETTINGS["yolo_conf"], verbose=False)[0]
    idx = vp._best_known_detection(res)
    if idx is not None and res.masks is not None:
      yolo_cls = vp.object_key(res.names[int(res.boxes.cls[idx])])
      if yolo_cls == key:
        m = res.masks.data[idx].cpu().numpy()
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
        if vp.mask_iou(m, true_mask) > 0.5:
          mask, box, src = m, [int(v) for v in res.boxes.xyxy[idx].cpu().numpy()], "yolo"
  if mask is None:
    mask = sim_yolo_mask(true_mask, rng)
    box = box_of(mask)
  row.update(mask_source=src, yolo_class=yolo_cls or "none",
             mask_iou_vs_true=round(vp.mask_iou(mask, true_mask), 3))
  obs = vp.Observation(color=bgr, depth=depth, mask=mask, box=box,
                       class_name=key, object_key=key, confidence=1.0, K=K)
  row["UD"] = round(vp.depth_incompleteness(obs), 3)
  T_gt = T_wc @ T_ow
  t0 = time.time()
  with contextlib.redirect_stdout(io.StringIO()):
    main, fb = class_checked(obs, cads)
  row["runtime_s"] = round(time.time() - t0, 1)
  for tag, o in (("main", main), ("fb", fb)):
    fin = o.get("final_object", key)
    e_r, e_t = pose_error(T_gt, o.get("T_cad_to_cam"), cads[key]) \
        if fin == key else (np.nan, np.nan)
    row.update({f"state_{tag}": o["state"], f"reason_{tag}": o.get("reason", ""),
                f"C_{tag}": round(o.get("C", 0.0), 3),
                f"chosen_{tag}": o.get("chosen", ""),
                f"class_{tag}": fin,
                f"rot_err_{tag}": round(e_r, 2) if e_r == e_r else "",
                f"trans_err_{tag}": round(e_t, 2) if e_t == e_t else ""})
  row["used_fallback"] = bool(fb.get("used_fallback"))
  return row


# ------------------------------------------------------------------ summary
def _f(x):
  try:
    return float(x)
  except (TypeError, ValueError):
    return np.nan


def classify(r, tag):
  if r.get("state_main") == "skip":
    return "skip"
  er, et = _f(r[f"rot_err_{tag}"]), _f(r[f"trans_err_{tag}"])
  wrong_class = r[f"class_{tag}"] != r["object"]
  good = (not wrong_class) and er <= SUCCESS_ROT and et <= SUCCESS_TRANS
  bad = wrong_class or not (er <= BAD_ROT and et <= BAD_TRANS)
  valid = r[f"state_{tag}"] == "Valid"
  if valid and good:
    return "valid_good"
  if valid and bad:
    return "FALSE_ACCEPT"
  if valid:
    return "valid_5to10"
  return "conflict_bad" if bad else "conflict_good"


def summarize(rows):
  lines = []
  rows = [r for r in rows if r.get("state_main") != "skip"]
  n = len(rows)
  lines.append(f"Trials: {n}   success = rot <= {SUCCESS_ROT} deg and trans <= "
               f"{SUCCESS_TRANS} mm (symmetry-aware); bad = > {BAD_ROT} deg or > "
               f"{BAD_TRANS} mm or wrong class")
  lines.append(f"Mask source: YOLO {sum(r['mask_source']=='yolo' for r in rows)}, "
               f"simulated {sum(r['mask_source']=='sim' for r in rows)}")
  for tag, title in (("main", "MAIN PATH (no table fallback)"),
                     ("fb", "WITH TABLE YAW-SWEEP FALLBACK")):
    lines.append(f"\n=== {title} ===")
    hdr = (f"{'group':28s} {'n':>4s} {'Valid':>6s} {'acc%':>6s} {'FA':>4s} "
           f"{'rejBad':>6s} {'rejGood':>7s} {'medRot':>7s} {'medTr':>6s}")
    lines.append(hdr)

    def block(name, rs):
      if not rs:
        return
      cl = [classify(r, tag) for r in rs]
      v = sum(c.startswith("valid") or c == "FALSE_ACCEPT" for c in cl)
      good_all = sum(c in ("valid_good", "conflict_good") for c in cl)
      good_valid = sum(c == "valid_good" for c in cl)
      fa = cl.count("FALSE_ACCEPT")
      rb = cl.count("conflict_bad")
      rg = cl.count("conflict_good")
      er = [_f(r[f"rot_err_{tag}"]) for r in rs if r[f"state_{tag}"] == "Valid"]
      et = [_f(r[f"trans_err_{tag}"]) for r in rs if r[f"state_{tag}"] == "Valid"]
      lines.append(f"{name:28s} {len(rs):4d} {v:6d} {100*good_valid/len(rs):6.1f} "
                   f"{fa:4d} {rb:6d} {rg:7d} {np.nanmedian(er) if er else np.nan:7.2f} "
                   f"{np.nanmedian(et) if et else np.nan:6.2f}")

    block("ALL", rows)
    for k in MESHES:
      block(f"obj={k}", [r for r in rows if r["object"] == k])
    for c in COLORS:
      block(f"color={c}", [r for r in rows if r["color"] == c])
    for s in SCENARIOS:
      block(f"scenario={s}", [r for r in rows if r["scenario"] == s])
    for t in TILTS:
      block(f"view={t}deg", [r for r in rows if int(r["tilt"]) == t])
    for k in MESHES:
      for s in SCENARIOS:
        block(f"{k}/{s}", [r for r in rows if r["object"] == k and r["scenario"] == s])
  lines.append("\nColumns: Valid = gate accepted; acc% = accepted AND within success "
               "limits, as % of all trials; FA = false accepts (Valid but bad pose); "
               "rejBad = correctly rejected; rejGood = rejected although pose was "
               "fine; medRot/medTr = median error of accepted poses.")
  fas = [r for r in rows if classify(r, "main") == "FALSE_ACCEPT"
         or classify(r, "fb") == "FALSE_ACCEPT"]
  if fas:
    lines.append("\nFalse accepts (either path):")
    for r in fas:
      lines.append(f"  {r['object']}/{r['color']}/{r['scenario']} view {r['tilt']} "
                   f"prop_tilt {r['prop_tilt']}: main {r['state_main']} rot "
                   f"{r['rot_err_main']} tr {r['trans_err_main']} C {r['C_main']} | "
                   f"fb {r['state_fb']} rot {r['rot_err_fb']} tr {r['trans_err_fb']} "
                   f"C {r['C_fb']} fallback={r['used_fallback']}")
  return "\n".join(lines)


def main():
  workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
  reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
  only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(MESHES)
  os.makedirs(OUT, exist_ok=True)
  jobs, seed = [], 1000
  for k in only:
    for c in COLORS:
      for s in SCENARIOS:
        for t in TILTS:
          for rep in range(reps):
            seed += 1
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
  s = summarize(rows)
  with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write(s + "\n")
  print(s)


if __name__ == "__main__":
  main()
