"""V-GAP per-object pose pipeline (RealSense + YOLO26s-seg + Open3D).

Built from IOU_pipeline.py. Same capture flow (live YOLO preview, press 's' to
grab a 25-frame burst), but every stage is its own function and every
function reads its settings from the detected object's entry in
OBJECT_CONFIGS:

  capture_observation()  RealSense burst + YOLO class / box / mask
  build_point_cloud()    mask crop -> voxel -> outliers -> normals
  compute_fpfh()         FPFH descriptors
  ransac_register()      global registration (FPFH RANSAC, or sphere fit
                         for the hemisphere; table yaw sweep fallback)
  icp_refine()           multi-scale ICP (point-to-plane or point-to-point per stage)
  iou_check()            CAD projection, IB / IM / C, symmetry-aware
                         hypothesis test, Valid / Conflict gate
  estimate_pose_same_view()  one fresh burst from the same view on a
                         Conflict (SAME_VIEW_RETRIES) before any arm move

Conventions
  * Units are meters everywhere.
  * Registration runs scan -> CAD (as in IOU_pipeline.py), so every scan
    point has a CAD partner. The reported pose is T_cad_to_cam = inv(T_scan_to_cad).
  * The CAD cloud should be exterior-only (an "inside removed" STL).
    load_cad_model() strips interior surfaces automatically when given a mesh.

Plan equations: C = wB*IB + wM*IM (Eq. 5-7), gate (Eq. 8), UD (Eq. 9),
Ulive (Eq. 10), symmetric rotation error (Eq. 18).
"""

import copy
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d

DEBUG_VIEWS = False  # True opens the Open3D / OpenCV windows at each stage

# ==============================================================================
# GLOBAL SETTINGS (shared by all objects; the plan keeps these fixed)
# ==============================================================================
CAMERA_SETTINGS = dict(
    width=640, height=480, fps=30,
    laser_power=150.0,       # max emitter power, as in IOU_pipeline.py changed from 360.0 to 150.0
    exposure_us=4000.0,     # manual depth exposure, as in IOU_pipeline.py changed from 16,000 to 4000
    frames_to_capture=25,
    max_depth_m=0.8,
    yolo_conf=0.40,       # value in the 10/22 gate runs' result.json (was 0.60)
)
YOLO_MODEL_PATH = "../Python Files Disassembly/runs/segment/yolo26s_ft_nonull/weights/best.pt"   # no-null fine-tune of YOLO26s-seg (the deployed model); must be a -seg model (masks needed). Path is relative to V-GAP/ in the lab repo.

# 3D check (after C and the margin): share of scan points within F3_DIST of
# the posed CAD must be >= f3_min (per object). C only compares outlines, so
# it cannot see a pose that is shifted in depth or tilted with the same
# outline; this catches those. Thresholds from the 10/22 gate runs (correct
# poses average 0.92-0.97). False = C and margin only, as in those runs.
CHECK_3D = True

# Same-view retry: when YOLO found the object but the gate says Conflict,
# take one fresh burst from the same camera pose before any arm move (a new
# capture and a new RANSAC draw, ~3.5 s). In the 10/22 gate runs the next
# burst at the same view passed 26 of 61 times after an L / T / gearbox
# Conflict (25 correct, 1 a bit off, 0 wrong). No detection, or a
# second Conflict, goes straight to the recovery moves. 0 = no retry.
SAME_VIEW_RETRIES = 1
F3_DIST = 0.003

# Detection filters (run on every YOLO frame before a detection is picked).
# Gripper mask: a piece of the gripper sits in the top-left corner of the
# wrist camera's 640x480 color image in every frame, and YOLO sometimes calls
# it a gearbox (conf 0.5-0.7). A detection whose mask lies mostly inside this
# box (x1, y1, x2, y2 px, grown by GRIPPER_MARGIN_PX) is dropped. None = off.
GRIPPER_ROI = (0, 0, 115, 95)
GRIPPER_MARGIN_PX = 10
GRIPPER_MAX_INSIDE = 0.5     # drop when more than this share of the mask is inside
# Nearest-to-target: when the object's base-frame position is known (gate
# eval --target, or found at the start), keep the detection whose mask center
# is nearest to it in the table plane, and drop it if it is farther than
# TARGET_MAX_DIST_M (0 = keep the nearest whatever the distance). Occluders
# and distractors YOLO labels as a V-GAP class are thrown out this way unless
# they sit right on the object; then the gate has to reject them.
TARGET_MAX_DIST_M = 0.10      # 5 cm cut the real gearbox at tilted views (66 mm off: mask center + hand-eye error)

# ==============================================================================
# PER-OBJECT PARAMETERS
# ==============================================================================
# Sizing rules used below (depth noise of a D435 at 0.4-0.5 m is ~1 mm and one
# pixel covers ~0.75 mm, so nothing below ~1.5 mm is worth resolving):
#   voxel        ~1/40 to 1/60 of the longest side, and at least 3-4 voxels
#                across the thinnest feature that must survive
#   normals      2-2.5 x voxel (enough neighbours, still sharp at edges)
#   FPFH radius  5 x voxel; must stay below the smallest distinctive feature
#                width so descriptors stay local to corners and lips
#   RANSAC dist  1.5 x voxel
#   ICP          3 stages, roughly 5x -> 3x -> 1.5x voxel
#   tau_c, delta from the gearbox simulation (good poses C >= 0.88, bad
#                <= 0.77); recalibrate per object in week 5.
OBJECT_CONFIGS = {
    # Assembled gearbox 153 x 90 x 64.8 mm (lip to lip). Body without lips
    # 111.6 x 90 (only 1.24:1, so a 90 deg yaw flip fits almost as well).
    # Lips: 9 mm thick, stick out 20.7 mm. Exterior is D2-symmetric, so
    # 180 deg flips are harmless and only the 90 deg yaw needs testing.
    "gearbox": dict(
        cad_path="cad/gearbox/gearbox_assembled_153mm.stl",   # 153 x 90 x 64.8 mm
        cad_units="mm",
        expected_extent_mm=(153.0, 90.0, 64.8),
        voxel=0.0025,              # your tuned value; lip = 3.6 voxels thick
        mask_erode_px=2,
        outlier_nb=20, outlier_std=2.0,   # looser than 1.5 so lip edges survive
        normal_radius=0.005,
        fpfh_radius=0.0125, fpfh_max_nn=100,
        global_method="fpfh_ransac",
        ransac_dist=0.00375, ransac_edge=0.9, ransac_normal_deg=45.0,
        ransac_iters=1_000_000, ransac_conf=0.999,
        icp_thresholds=(0.012, 0.008, 0.004), icp_iters=(100, 60, 40),
        # point-to-plane slides along the big flat top when that is all the
        # camera sees, so the last stage is point-to-point (uses the outline)
        icp_methods=("plane", "plane", "point"),
        symmetry="auto",            # finds the D2 flips from the CAD
        hypotheses=[("height", 90)],
        fallback="table_yaw_sweep", sweep_step_deg=15, sweep_keep=3,
        table_ring_px=(8, 30),
        wB=0.3, wM=0.7, tau_c=0.90, delta=0.10, f3_min=0.7,
    ),
    # L block 100 x 100 x 26.7 mm, arms 33.3 mm wide. Only proper symmetry:
    # 180 deg about the corner diagonal (flips it over onto the same L).
    # FPFH at 10 mm sits well inside the 33 mm arm, so features stay local
    # to the inner and outer corners.
    "l_block": dict(
        cad_path="cad/l_block/l_block_100mm_reference.stl",
        cad_units="mm",
        expected_extent_mm=(100.0, 100.0, 26.7),
        voxel=0.002,               # 13 voxels through the height
        mask_erode_px=2,
        outlier_nb=20, outlier_std=2.0,
        normal_radius=0.005,
        fpfh_radius=0.010, fpfh_max_nn=100,
        global_method="fpfh_ransac",
        ransac_dist=0.003, ransac_edge=0.9, ransac_normal_deg=45.0,
        ransac_iters=1_000_000, ransac_conf=0.999,
        icp_thresholds=(0.010, 0.006, 0.003), icp_iters=(100, 60, 40),
        # a flat prism seen from above is one plane; point-to-plane ICP
        # cannot fix in-plane slide/rotation there, point-to-point can
        icp_methods=("point", "point", "point"),
        symmetry="auto",
        # big flat faces make 90/180/270 deg in-plane mistakes the likely
        # RANSAC failure; the silhouette tells them apart cheaply
        hypotheses=[("height", 90), ("height", 180), ("height", 270)],
        fallback="table_yaw_sweep", sweep_step_deg=15, sweep_keep=3,
        table_ring_px=(8, 30),
        wB=0.3, wM=0.7, tau_c=0.85, delta=0.10, f3_min=0.7,
    ),
    # T block 100 x 100 x 26.7 mm, crossbar and stem 33.3 mm. Only proper
    # symmetry: 180 deg about the stem axis.
    "t_block": dict(
        cad_path="cad/t_block/t_block_100mm_reference.stl",
        cad_units="mm",
        expected_extent_mm=(100.0, 100.0, 26.7),
        voxel=0.002,
        mask_erode_px=2,
        outlier_nb=20, outlier_std=2.0,
        normal_radius=0.005,
        fpfh_radius=0.010, fpfh_max_nn=100,
        global_method="fpfh_ransac",
        ransac_dist=0.003, ransac_edge=0.9, ransac_normal_deg=45.0,
        ransac_iters=1_000_000, ransac_conf=0.999,
        icp_thresholds=(0.010, 0.006, 0.003), icp_iters=(100, 60, 40),
        icp_methods=("point", "point", "point"),
        symmetry="auto",
        hypotheses=[("height", 90), ("height", 180), ("height", 270)],
        fallback="table_yaw_sweep", sweep_step_deg=15, sweep_keep=3,
        table_ring_px=(8, 30),
        wB=0.3, wM=0.7, tau_c=0.81, delta=0.10, f3_min=0.7,
    ),
    # Hemisphere D 100 x 50 mm, flat face down. Every patch of a sphere has
    # the same FPFH, so feature RANSAC is meaningless here. Instead: fit a
    # sphere of known radius to the dome (its center IS the base center),
    # and take the axis from the table plane around the mask. Yaw is
    # unobservable without the optional tab/notch, so it is a continuous
    # symmetry and Eq. 18 reduces to tilt error. ICP is skipped: the sphere
    # fit is already the least-squares fit and ICP on a dome can only
    # slide around the center.
    "hemisphere": dict(
        cad_path="cad/hemisphere/hemisphere_100mm_reference.stl",
        cad_units="mm",
        expected_extent_mm=(100.0, 100.0, 50.0),
        voxel=0.002,
        mask_erode_px=2,
        outlier_nb=20, outlier_std=2.0,
        normal_radius=0.006,
        fpfh_radius=0.010, fpfh_max_nn=100,   # unused (sphere_fit)
        global_method="sphere_fit",
        sphere_radius=0.050, sphere_inlier=0.002, sphere_iters=3000,
        # axis from the object (rim / flat base in the depth image), searched
        # within this cone around the table normal; False = table normal only
        axis_from_object=True, axis_max_tilt_deg=30.0, axis_tol_mm=4.0,
        axis_min_gain=0.01, axis_ambig_eps=0.002, axis_ambig_deg=10.0,
        table_ring_px=(8, 30),     # ring around the mask used for the table
        icp_thresholds=(), icp_iters=(), icp_methods=(),
        symmetry="axial",
        hypotheses=[],
        fallback=None,
        wB=0.3, wM=0.7, tau_c=0.85, delta=0.10,
        f3_min=0.6,
    ),
}

# Map YOLO class names to the keys above. Edit to match your model's names
# (print(yolo_model.names) to see them).
CLASS_ALIASES = {
    "gearbox": "gearbox", "gear_box": "gearbox",
    "l_block": "l_block", "l_shape": "l_block", "l_shaped": "l_block",
    "l": "l_block",
    "t_block": "t_block", "t_shape": "t_block", "t_shaped": "t_block",
    "t": "t_block",
    "hemisphere": "hemisphere", "dome": "hemisphere", "hemi": "hemisphere",
    "sphere": "hemisphere", "semi_sphere": "hemisphere",
}


def object_key(class_name):
  k = class_name.strip().lower().replace("-", "_").replace(" ", "_")
  return CLASS_ALIASES.get(k)


# Classes YOLO confuses with each other. When it reports one of these, the
# pose is also estimated with each rival's CAD against the same mask and
# depth, and the CAD with the higher C wins (see estimate_pose_with_class_check).
CLASS_RIVALS = {
    "l_block": ["t_block"],
    "t_block": ["l_block"],
}
# If the winning CAD leads the runner-up by less than this, the class is
# ambiguous and the result is a Conflict.
CLASS_DELTA = 0.10


# ==============================================================================
# DATA CONTAINERS
# ==============================================================================
@dataclass
class Observation:
  color: np.ndarray            # BGR image at the trigger moment
  depth: np.ndarray            # uint16 mm, burst median + bilateral filter
  mask: np.ndarray             # bool HxW, YOLO segmentation (M2D)
  box: list                    # [x1, y1, x2, y2] (B2D)
  class_name: str
  object_key: str
  confidence: float
  K: np.ndarray                # 3x3 color intrinsics (depth is aligned)
  meta: dict = field(default_factory=dict)


@dataclass
class CadModel:
  pcd: o3d.geometry.PointCloud         # voxelized, with outward normals
  dense_points: np.ndarray             # for silhouette projection
  center: np.ndarray                   # bounding-box center (CAD frame)
  axes: np.ndarray                     # columns: long, mid, height axis
  symmetries: list                     # 4x4 transforms in the CAD frame
  axial_axis: np.ndarray = None        # set for continuous symmetry
  base_center: np.ndarray = None       # hemisphere only


# ==============================================================================
# 1. REALSENSE + YOLO
# ==============================================================================
def start_realsense(settings=CAMERA_SETTINGS):
  import pyrealsense2 as rs
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, settings["width"], settings["height"],
                       rs.format.bgr8, settings["fps"])
  config.enable_stream(rs.stream.depth, settings["width"], settings["height"],
                       rs.format.z16, settings["fps"])
  profile = pipeline.start(config)
  align = rs.align(rs.stream.color)
  intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])

  depth_sensor = profile.get_device().first_depth_sensor()
  if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, settings["laser_power"])
  if depth_sensor.supports(rs.option.enable_auto_exposure):
    depth_sensor.set_option(rs.option.enable_auto_exposure, False)
  if depth_sensor.supports(rs.option.exposure):
    depth_sensor.set_option(rs.option.exposure, settings["exposure_us"])
  time.sleep(1)
  print(f"[+] RealSense started, fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
        f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
  return pipeline, align, K


def temporal_median_ignore_zeros(depth_buffer, min_valid_frac=0.3):
  """Median over the burst using valid readings only.

  np.median counts 0 (invalid) as a value, so a pixel valid in fewer than
  half the frames becomes 0. Thin lips and dark/reflective prints flicker,
  so this keeps far more of them.
  """
  stack = np.stack(depth_buffer, axis=0).astype(np.float32)
  stack[stack == 0] = np.nan
  valid_frac = np.mean(~np.isnan(stack), axis=0)
  with warnings.catch_warnings():  # all-invalid pixels are expected
    warnings.simplefilter("ignore", RuntimeWarning)
    med = np.nanmedian(stack, axis=0)
  med[(valid_frac < min_valid_frac) | np.isnan(med)] = 0
  return med.astype(np.uint16)


def _best_known_detection(result):
  """Index of the most confident detection whose class we have a config for."""
  best, best_conf = None, -1.0
  for i in range(len(result.boxes)):
    name = result.names[int(result.boxes.cls[i])]
    conf = float(result.boxes.conf[i])
    if object_key(name) and conf > best_conf:
      best, best_conf = i, conf
  return best


def _det_mask(result, i, shape):
  """Detection i's mask resized to the image (bool), or its box if no masks."""
  h, w = shape
  if result.masks is not None:
    m = result.masks.data[i].cpu().numpy()
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
  m = np.zeros((h, w), bool)
  x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i].cpu().numpy()]
  m[max(y1, 0):y2, max(x1, 0):x2] = True
  return m


def _mask_center_base(mask, depth, K, T_base_cam, z_fallback):
  """Base-frame point under the mask center (median depth of the mask;
  z_fallback when the mask has no depth)."""
  ys, xs = np.nonzero(mask)
  u, v = np.median(xs), np.median(ys)
  d = depth[mask] if depth is not None else np.array([])
  d = d[d > 0]
  z = np.median(d) / 1000.0 if len(d) >= 20 else z_fallback
  p = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
  return T_base_cam[:3, :3] @ p + T_base_cam[:3, 3]


def select_detection(result, shape, depth=None, K=None, T_base_cam=None,
                     target=None, gripper_roi="default", margin=None,
                     max_dist=None):
  """Pick the detection to register, after the gripper and target filters.

  Known classes only. Drops detections mostly inside the gripper box, then,
  if target (base frame, m) and T_base_cam are given, keeps the detection
  nearest the target in x/y and drops it if farther than max_dist. Without
  a target: the most confident remaining detection (as before).
  Returns (idx or None, info); info is logged per burst.
  """
  roi = GRIPPER_ROI if gripper_roi == "default" else gripper_roi
  margin = GRIPPER_MARGIN_PX if margin is None else margin
  max_dist = TARGET_MAX_DIST_M if max_dist is None else max_dist
  info = dict(n_detections=len(result.boxes), n_dropped_gripper=0,
              n_dropped_far=0, n_dropped_unknown=0, target_dist_mm="",
              dropped="")
  dropped, keep = [], []
  for i in range(len(result.boxes)):
    name = result.names[int(result.boxes.cls[i])]
    conf = float(result.boxes.conf[i])
    tag = f"{name}@{conf:.2f}"
    if not object_key(name):
      info["n_dropped_unknown"] += 1
      dropped.append(f"{tag}:unknown class")
      continue
    m = _det_mask(result, i, shape)
    if not m.any():
      continue
    if roi is not None:
      x1, y1, x2, y2 = roi
      inside = m[max(y1 - margin, 0):y2 + margin, max(x1 - margin, 0):x2 + margin].sum()
      if inside / m.sum() > GRIPPER_MAX_INSIDE:
        info["n_dropped_gripper"] += 1
        dropped.append(f"{tag}:gripper")
        continue
    keep.append((i, conf, m, tag))

  best = None
  if target is not None and T_base_cam is not None and K is not None and keep:
    t = np.asarray(target, float)
    z_t = (np.linalg.inv(T_base_cam) @ np.append(t, 1.0))[2]
    dists = [np.linalg.norm(_mask_center_base(m, depth, K, T_base_cam, z_t)[:2] - t[:2])
             for _, _, m, _ in keep]
    order = np.argsort(dists)
    j = order[0]
    for k in order[1:]:
      dropped.append(f"{keep[k][3]}:not nearest ({1000 * dists[k]:.0f} mm)")
    info["n_dropped_far"] = len(keep) - 1
    if max_dist and dists[j] > max_dist:
      dropped.append(f"{keep[j][3]}:far ({1000 * dists[j]:.0f} mm)")
      info["n_dropped_far"] += 1
    else:
      best = keep[j][0]
      info["target_dist_mm"] = round(1000 * dists[j], 1)
  elif keep:
    best = max(keep, key=lambda k: k[1])[0]
  info["dropped"] = "; ".join(dropped)
  if dropped:
    print(f"[*] Detections dropped: {info['dropped']}")
  return best, info


def capture_observation(pipeline, align, K, yolo_model,
                        settings=CAMERA_SETTINGS):
  """Live preview; 's' captures a burst of the best known detection.

  Returns an Observation, or None if the user quits with 'q' / ESC.
  """
  print("\n[*] Live preview. 's' = capture burst, 'q' = quit")
  while True:
    frames = align.process(pipeline.wait_for_frames())
    depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
    if not depth_frame or not color_frame:
      continue
    depth = np.asanyarray(depth_frame.get_data()).copy()
    color = np.asanyarray(color_frame.get_data()).copy()
    result = yolo_model(color, conf=settings["yolo_conf"], agnostic_nms=True,
                        verbose=False)[0]
    cv2.imshow("Live Stream (s = capture)", result.plot())
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), 27):
      cv2.destroyAllWindows()
      return None
    if key not in (ord("s"), ord("S")):
      continue

    idx, info = select_detection(result, depth.shape, depth, K)
    if idx is None:
      print("[-] No known object detected in this frame.")
      continue
    cv2.destroyAllWindows()
    obs = _burst_observation(pipeline, align, K, result, idx, depth, color,
                             settings)
    obs.meta["det_filter"] = info
    return obs


def capture_observation_now(pipeline, align, K, yolo_model,
                            settings=CAMERA_SETTINGS, max_frames=15):
  """Same burst as capture_observation, with no key press (for the
  same-view retry). Returns None if YOLO finds no known object in
  max_frames frames, which sends the caller to the recovery moves."""
  for _ in range(max_frames):
    frames = align.process(pipeline.wait_for_frames())
    depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
    if not depth_frame or not color_frame:
      continue
    depth = np.asanyarray(depth_frame.get_data()).copy()
    color = np.asanyarray(color_frame.get_data()).copy()
    result = yolo_model(color, conf=settings["yolo_conf"], agnostic_nms=True,
                        verbose=False)[0]
    idx, info = select_detection(result, depth.shape, depth, K)
    if idx is not None:
      obs = _burst_observation(pipeline, align, K, result, idx, depth, color,
                               settings)
      obs.meta["det_filter"] = info
      return obs
  print("[-] No known object detected.")
  return None


def _burst_observation(pipeline, align, K, result, idx, depth, color, settings):
  """25-frame depth burst after a detection -> Observation."""
  if result.masks is None:
    raise RuntimeError("YOLO returned no masks; load the -seg model.")

  burst = [depth]
  while len(burst) < settings["frames_to_capture"]:
    b = align.process(pipeline.wait_for_frames()).get_depth_frame()
    if b:
      burst.append(np.asanyarray(b.get_data()).copy())

  h, w = depth.shape
  mask = result.masks.data[idx].cpu().numpy()
  mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
  box = [int(v) for v in result.boxes.xyxy[idx].cpu().numpy()]
  name = result.names[int(result.boxes.cls[idx])]

  median = temporal_median_ignore_zeros(burst)
  filt = cv2.bilateralFilter(median.astype(np.float32), d=5,
                             sigmaColor=15.0, sigmaSpace=5.0)
  filt[median == 0] = 0  # do not let the filter invent depth in holes
  print(f"[+] Captured '{name}' (conf {float(result.boxes.conf[idx]):.2f}).")
  return Observation(
      color=color, depth=filt.astype(np.uint16), mask=mask, box=box,
      class_name=name, object_key=object_key(name),
      confidence=float(result.boxes.conf[idx]), K=K,
      meta=dict(settings=settings,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")))


# ==============================================================================
# CAD MODEL (loaded once per object)
# ==============================================================================
def _rot(axis, deg):
  return o3d.geometry.get_rotation_matrix_from_axis_angle(
      np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))


def _rot_about(center, axis, deg):
  T = np.eye(4)
  T[:3, :3] = _rot(axis, deg)
  T[:3, 3] = center - T[:3, :3] @ center
  return T


def principal_frame(points, snap_deg=10.0):
  """Bounding-box center and PCA axes (long, mid, height), snapped to the
  CAD's x/y/z when within snap_deg; CAD exports are normally axis-aligned
  and a few degrees of PCA tilt would make symmetric flips look wrong."""
  center = 0.5 * (points.min(axis=0) + points.max(axis=0))
  _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
  axes = vt.T.copy()
  for i in range(3):
    j = int(np.argmax(np.abs(axes[:, i])))
    if abs(axes[j, i]) >= np.cos(np.radians(snap_deg)):
      axes[:, i] = np.sign(axes[j, i]) * np.eye(3)[:, j]
  if np.linalg.det(axes) < 0:
    axes[:, 2] *= -1
  return center, axes


def _exterior_samples(mesh, n_points):
  """Samples the mesh and keeps points visible from outside."""
  mesh.compute_triangle_normals()
  pcd = mesh.sample_points_uniformly(n_points, use_triangle_normal=True)
  pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  eps = 1e-5 * np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
  rays = np.hstack([pts + nrm * eps, nrm]).astype(np.float32)
  hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())
  out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[~hit]))
  out.normals = o3d.utility.Vector3dVector(nrm[~hit])
  return out


def _symmetry_error(points, tree, T):
  q = (T[:3, :3] @ points.T).T + T[:3, 3]
  d = np.array([np.sqrt(tree.search_knn_vector_3d(p, 1)[2][0]) for p in q])
  return float(np.percentile(d, 95))  # median would pass partial overlaps


def detect_symmetries(points, center, axes, tol):
  """Finds proper rotations (90/180 deg about the principal axes and 180 deg
  about the in-plane diagonals) that map the exterior cloud onto itself."""
  sub = points[np.random.default_rng(0).choice(
      len(points), min(3000, len(points)), replace=False)]
  pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
  tree = o3d.geometry.KDTreeFlann(pc)
  cands = []
  for i in range(3):
    cands += [(axes[:, i], 90), (axes[:, i], 180), (axes[:, i], 270)]
  for sgn in (1, -1):  # in-plane diagonals (the L block's symmetry axis)
    cands.append((axes[:, 0] + sgn * axes[:, 1], 180))
  found = [np.eye(4)]
  for axis, deg in cands:
    T = _rot_about(center, axis, deg)
    if _symmetry_error(sub, tree, T) < tol:
      found.append(T)
  return found


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
  """Finds a relative path next to this script first, then in the folder
  you ran it from. Raises a clear error if the file is in neither."""
  if os.path.isabs(path):
    candidates = [path]
  else:
    candidates = [os.path.join(SCRIPT_DIR, path), os.path.abspath(path)]
  for c in candidates:
    if os.path.isfile(c):
      return c
  raise FileNotFoundError(
      f"CAD file not found: '{path}'. Looked in:\n  " + "\n  ".join(candidates)
      + "\nCopy the file there or set cad_path in OBJECT_CONFIGS.")


def load_cad_model(params, n_dense=60000):
  """Loads the CAD as an exterior-only cloud in meters, checks its size and
  works out its symmetries."""
  path = resolve_path(params["cad_path"])
  scale = 0.001 if params["cad_units"] == "mm" else 1.0
  if path.lower().endswith((".pcd", ".ply")):
    dense = o3d.io.read_point_cloud(path)
    dense.scale(scale, center=np.zeros(3))
  else:
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.scale(scale, center=np.zeros(3))
    dense = _exterior_samples(mesh, n_dense)

  pts = np.asarray(dense.points)
  ext = np.sort(pts.max(axis=0) - pts.min(axis=0))[::-1] * 1000
  exp = np.sort(np.asarray(params["expected_extent_mm"]))[::-1]
  if np.any(np.abs(ext - exp) / exp > 0.05):
    print(f"[!] WARNING: CAD extent {np.round(ext,1)} mm does not match the "
          f"expected {exp} mm. Check units/scale of {path}.")

  pcd = dense.voxel_down_sample(params["voxel"])
  if not pcd.has_normals():
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=params["normal_radius"], max_nn=30))
  # outward normals (all four objects are convex enough about their center)
  c = pts.mean(axis=0)
  n, p = np.asarray(pcd.normals), np.asarray(pcd.points)
  n[np.sum(n * (p - c), axis=1) < 0] *= -1
  pcd.normals = o3d.utility.Vector3dVector(n)

  center, axes = principal_frame(pts)
  model = CadModel(pcd=pcd, dense_points=pts, center=center, axes=axes,
                   symmetries=[np.eye(4)])
  if params["symmetry"] == "auto":
    model.symmetries = detect_symmetries(pts, center, axes,
                                         tol=2 * params["voxel"])
  elif params["symmetry"] == "axial":
    # hemisphere: shortest extent is the axis; the flat base is the side
    # with many points on the bounding plane
    ext_xyz = pts.max(axis=0) - pts.min(axis=0)
    up_i = int(np.argmin(ext_xyz))
    lo = np.sum(pts[:, up_i] < pts[:, up_i].min() + 0.001)
    hi = np.sum(pts[:, up_i] > pts[:, up_i].max() - 0.001)
    up = np.eye(3)[:, up_i] * (1 if lo > hi else -1)
    base = center.copy()
    base[up_i] = pts[:, up_i].min() if lo > hi else pts[:, up_i].max()
    model.axial_axis, model.base_center = up, base
  print(f"[+] CAD loaded: {len(pcd.points)} pts, extent {np.round(ext,1)} mm, "
        f"{len(model.symmetries)} rotational symmetries"
        f"{' + axial' if model.axial_axis is not None else ''}")
  return model


# ==============================================================================
# 2. POINT CLOUD
# ==============================================================================
def build_point_cloud(obs, params, settings=CAMERA_SETTINGS):
  """YOLO mask crop (no table, no background) -> voxel -> outliers -> normals."""
  mask = obs.mask
  if params["mask_erode_px"] > 0:
    k = 2 * params["mask_erode_px"] + 1
    mask = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
  depth = np.where(mask, obs.depth, 0).astype(np.uint16)
  h, w = depth.shape
  intr = o3d.camera.PinholeCameraIntrinsic(w, h, obs.K[0, 0], obs.K[1, 1],
                                           obs.K[0, 2], obs.K[1, 2])
  pcd = o3d.geometry.PointCloud.create_from_depth_image(
      o3d.geometry.Image(depth), intr, depth_scale=1000.0,
      depth_trunc=settings["max_depth_m"])
  n_raw = len(pcd.points)
  pcd = pcd.voxel_down_sample(params["voxel"])
  if len(pcd.points) > params["outlier_nb"]:
    pcd, _ = pcd.remove_statistical_outlier(params["outlier_nb"],
                                            params["outlier_std"])
  pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
      radius=params["normal_radius"], max_nn=30))
  pcd.orient_normals_towards_camera_location(np.zeros(3))
  print(f"[+] Scan cloud: {n_raw} raw -> {len(pcd.points)} points")
  if DEBUG_VIEWS:
    o3d.visualization.draw_geometries([pcd], window_name="Scan cloud",
                                      point_show_normal=True)
  return pcd


def depth_incompleteness(obs):
  """UD = 1 - Nvalid / Nmask (Eq. 9)."""
  n = int(obs.mask.sum())
  return 1.0 if n == 0 else 1.0 - float(((obs.depth > 0) & obs.mask).sum()) / n


# ==============================================================================
# 3. FPFH
# ==============================================================================
def compute_fpfh(pcd, params):
  return o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(
          radius=params["fpfh_radius"], max_nn=params["fpfh_max_nn"]))


# ==============================================================================
# 4. GLOBAL REGISTRATION (RANSAC)
# ==============================================================================
def ransac_register(scan, cad, params, scan_fpfh=None, cad_fpfh=None, obs=None):
  """Returns T_scan_to_cad.

  fpfh_ransac: feature RANSAC (gearbox, L, T).
  sphere_fit : hemisphere, known-radius sphere RANSAC + table normal.
  """
  if params["global_method"] == "sphere_fit":
    return _hemisphere_pose(scan, cad, params, obs)

  reg = o3d.pipelines.registration
  d = params["ransac_dist"]
  result = reg.registration_ransac_based_on_feature_matching(
      scan, cad.pcd, scan_fpfh, cad_fpfh, mutual_filter=True,
      max_correspondence_distance=d,
      estimation_method=reg.TransformationEstimationPointToPoint(False),
      ransac_n=3,
      checkers=[reg.CorrespondenceCheckerBasedOnEdgeLength(params["ransac_edge"]),
                reg.CorrespondenceCheckerBasedOnDistance(d),
                reg.CorrespondenceCheckerBasedOnNormal(
                    np.radians(params["ransac_normal_deg"]))],
      criteria=reg.RANSACConvergenceCriteria(params["ransac_iters"],
                                             params["ransac_conf"]))
  print(f"[+] RANSAC fitness {result.fitness:.3f}, rmse "
        f"{result.inlier_rmse*1000:.2f} mm")
  return result.transformation


def _fit_sphere_known_radius(pts, r, inlier, iters, rng):
  """RANSAC on 4-point sphere fits, keeping those near radius r, then a
  least-squares refinement of the center with r fixed."""
  best_c, best_n = None, -1
  for _ in range(iters):
    s = pts[rng.choice(len(pts), 4, replace=False)]
    A = np.hstack([2 * s, np.ones((4, 1))])
    try:
      sol = np.linalg.solve(A, np.sum(s * s, axis=1))
    except np.linalg.LinAlgError:
      continue
    c = sol[:3]
    rr = np.sqrt(max(sol[3] + c @ c, 0))
    if abs(rr - r) > 3 * inlier:
      continue
    n = np.sum(np.abs(np.linalg.norm(pts - c, axis=1) - r) < inlier)
    if n > best_n:
      best_c, best_n = c, n
  if best_c is None:
    raise RuntimeError("Sphere fit failed: no candidate near the known radius.")
  c = best_c
  for _ in range(20):  # Gauss-Newton on |p - c| = r over inliers
    v = pts - c
    dist = np.linalg.norm(v, axis=1)
    inl = np.abs(dist - r) < inlier
    J = -v[inl] / dist[inl, None]
    res = dist[inl] - r
    c = c - np.linalg.lstsq(J, res, rcond=None)[0]
  return c, int(inl.sum())


def _table_normal(obs, params, settings=CAMERA_SETTINGS):
  """Plane through the depth ring just outside the object mask."""
  r_in, r_out = params["table_ring_px"]
  m = obs.mask.astype(np.uint8)
  ring = (cv2.dilate(m, np.ones((2 * r_out + 1,) * 2, np.uint8)) > 0) & \
         ~(cv2.dilate(m, np.ones((2 * r_in + 1,) * 2, np.uint8)) > 0)
  depth = np.where(ring, obs.depth, 0).astype(np.uint16)
  h, w = depth.shape
  intr = o3d.camera.PinholeCameraIntrinsic(w, h, obs.K[0, 0], obs.K[1, 1],
                                           obs.K[0, 2], obs.K[1, 2])
  pc = o3d.geometry.PointCloud.create_from_depth_image(
      o3d.geometry.Image(depth), intr, depth_scale=1000.0,
      depth_trunc=settings["max_depth_m"])
  if len(pc.points) < 50:
    return None
  plane, _ = pc.segment_plane(0.003, 3, 500)
  n = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
  centroid = np.asarray(pc.points).mean(axis=0)
  return n if n @ (-centroid) > 0 else -n  # point toward the camera


def _rotation_a_to_b(a, b):
  a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
  v, c = np.cross(a, b), float(a @ b)
  if np.linalg.norm(v) < 1e-9:
    return np.eye(3) if c > 0 else _rot(np.cross(a, [1, 0, 0]) if abs(a[0]) < .9
                                        else np.cross(a, [0, 1, 0]), 180)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx / (1 + c)


def _hemisphere_depth(dirs, center, axis, r):
  """Camera-frame depth (z) of a hemisphere (sphere centre = base centre
  `center`, dome on the +axis side, flat base disc) along unit rays `dirs`
  from the camera origin. NaN where a ray misses."""
  b = dirs @ center
  disc = b * b - (center @ center - r * r)
  t = np.full(len(dirs), np.inf)
  ok = disc >= 0
  sq = np.sqrt(np.where(ok, disc, 0))
  for tt in (b - sq, b + sq):            # dome: nearest sphere hit on +axis side
    p_side = (tt[:, None] * dirs - center) @ axis
    hit = ok & (tt > 0) & (p_side >= 0) & (tt < t)
    t[hit] = tt[hit]
  da = dirs @ axis                       # flat base disc
  with np.errstate(divide="ignore", invalid="ignore"):
    td = (center @ axis) / da
  pd = td[:, None] * dirs - center
  hit = (td > 0) & (np.sum(pd * pd, axis=1) <= r * r) & (td < t)
  t[hit] = td[hit]
  z = t * dirs[:, 2]
  z[~np.isfinite(t)] = np.nan
  return z


def _hemisphere_axis_search(obs, center, prior, params, height=None):
  """Finds the hemisphere axis from the object itself.

  The sphere fit fixes the centre but not which way the flat base faces. A
  tilted hemisphere differs from an upright one only where the dome ends at
  the rim, and where the flat base comes into view, so: every candidate axis
  within `axis_max_tilt_deg` of `prior` (the table normal) is rendered as a
  depth image (exact ray/sphere + ray/disc) and compared with the measured
  depth around the mask. Score = pixels where both agree within `axis_tol_mm`
  divided by (agree + predicted-object-but-farther-surface-measured +
  object-pixel-unexplained). Pixels where something nearer than the
  prediction was measured (an occluder) are ignored; "object pixels" are
  mask pixels whose 3D point lies inside the fitted ball and above the
  base level, so table pixels in a generous mask do not count. If `height`
  (centre above the table, m) is known, tilts that would push the base
  below the table are skipped: the rim's lowest point sits r*sin(tilt)
  below the centre. Coarse grid, then a fine grid around the best; the
  prior is kept unless the best tilt scores `axis_min_gain` higher.
  Returns (axis, tilt from prior in deg, score, ambiguity in deg).
  """
  r = params["sphere_radius"]
  tol = params.get("axis_tol_mm", 4.0) / 1000.0
  max_tilt = params.get("axis_max_tilt_deg", 30.0)
  h, w = obs.depth.shape
  ys, xs = np.nonzero(obs.mask)
  pad = 15
  y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
  x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
  vv, uu = np.mgrid[y0:y1, x0:x1]
  zm = obs.depth[y0:y1, x0:x1].astype(float).ravel() / 1000.0
  m = obs.mask[y0:y1, x0:x1].ravel()
  valid = zm > 0
  d = np.stack([(uu.ravel() - obs.K[0, 2]) / obs.K[0, 0],
                (vv.ravel() - obs.K[1, 2]) / obs.K[1, 1],
                np.ones(uu.size)], 1)
  d /= np.linalg.norm(d, axis=1, keepdims=True)
  zm_ray = zm  # depth images store z, and so does _hemisphere_depth
  P = d * (zm / np.where(d[:, 2] > 0, d[:, 2], 1))[:, None]
  prior = prior / np.linalg.norm(prior)
  rel = P - center
  objlike = m & valid & (np.linalg.norm(rel, axis=1) < r + tol) & \
      (rel @ prior > tol)
  max_tilt_table = max_tilt
  if height is not None:
    s_max = (height + tol) / r
    max_tilt_table = 90.0 if s_max >= 1 else float(np.degrees(np.arcsin(max(s_max, 0))))
  max_tilt = min(max_tilt, max_tilt_table)

  def score(a):
    zp = _hemisphere_depth(d, center, a, r)
    pred = np.isfinite(zp)
    agree = pred & valid & (np.abs(zp - zm_ray) < tol)
    beyond = pred & valid & (zm_ray > zp + tol)     # saw past the object
    unexpl = ~pred & objlike                        # object pixel, no object
    return agree.sum() / max(1, agree.sum() + beyond.sum() + unexpl.sum())

  ref = np.cross(prior, [1.0, 0, 0])
  if np.linalg.norm(ref) < 0.1:
    ref = np.cross(prior, [0, 1.0, 0])
  ref /= np.linalg.norm(ref)

  def ring(base, tilt, step_az):
    out = []
    e1 = np.cross(base, ref)
    if np.linalg.norm(e1) < 1e-6:
      e1 = np.cross(base, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    for az in np.arange(0, 360, step_az):
      out.append(_rot(base, az) @ (_rot(e1, tilt) @ base))
    return out

  cands = [prior]
  for tilt in np.arange(3, max_tilt + 1e-6, 3):
    n_az = max(6, int(round(360 * np.sin(np.radians(tilt)) / 3)))
    cands += ring(prior, tilt, 360 / n_az)
  scores = [score(a) for a in cands]
  best = cands[int(np.argmax(scores))]
  fine = [best]
  for tilt in (1, 2):
    fine += ring(best, tilt, 45 if tilt == 1 else 30)
  fine = [a for a in fine
          if np.degrees(np.arccos(np.clip(a @ prior, -1, 1))) <= max_tilt + 1e-6]
  fscores = [score(a) for a in fine]
  a = fine[int(np.argmax(fscores))]
  # small tilts barely change the depth image, so stay on the table normal
  # unless a tilted axis explains the depth clearly better
  if max(fscores) - scores[0] < params.get("axis_min_gain", 0.01):
    a = prior
  # ambiguity: how far from the chosen axis another axis still explains the
  # depth almost as well. Large when the dome faces the camera and the rim
  # and base are out of sight; then the tilt is unobservable from this view.
  top = max(max(scores), max(fscores))
  eps = params.get("axis_ambig_eps", 0.002)
  ambig = max(float(np.degrees(np.arccos(np.clip(c @ a, -1, 1))))
              for c, sc in zip(cands + fine, scores + fscores) if sc >= top - eps)
  tilt = float(np.degrees(np.arccos(np.clip(a @ prior, -1, 1))))
  return a, tilt, float(max(fscores)), ambig


def _height_above_table(obs, params, n, point, settings=CAMERA_SETTINGS):
  """Height of `point` above the table plane (normal n) from the depth ring
  just outside the mask, or None if the ring has too few points."""
  r_in, r_out = params["table_ring_px"]
  m = obs.mask.astype(np.uint8)
  ring = (cv2.dilate(m, np.ones((2 * r_out + 1,) * 2, np.uint8)) > 0) & \
         ~(cv2.dilate(m, np.ones((2 * r_in + 1,) * 2, np.uint8)) > 0) & \
         (obs.depth > 0) & (obs.depth < settings["max_depth_m"] * 1000)
  vv, uu = np.nonzero(ring)
  if len(vv) < 50:
    return None
  z = obs.depth[vv, uu] / 1000.0
  pts = np.stack([(uu - obs.K[0, 2]) * z / obs.K[0, 0],
                  (vv - obs.K[1, 2]) * z / obs.K[1, 1], z], 1)
  return float(np.median((point - pts) @ n))


def _hemisphere_pose(scan, cad, params, obs):
  pts = np.asarray(scan.points)
  center, n_in = _fit_sphere_known_radius(
      pts, params["sphere_radius"], params["sphere_inlier"],
      params["sphere_iters"], np.random.default_rng(0))
  up = _table_normal(obs, params) if obs is not None else None
  height = None
  if up is None:
    # fallback: dome points sit above the base, so their mean is "up"
    up = pts.mean(axis=0) - center
    up /= np.linalg.norm(up)
    print("[!] Table ring had too few points; axis prior taken from the dome.")
  elif obs is not None:
    height = _height_above_table(obs, params, up, center)
  if obs is not None and params.get("axis_from_object", True):
    # the table normal is only the starting guess; the axis itself comes
    # from where the dome ends (rim) and whether the flat base shows
    up, tilt, s, ambig = _hemisphere_axis_search(obs, center, up, params, height)
    obs.meta["axis_ambiguity_deg"] = ambig
    print(f"[+] Hemisphere axis: {tilt:.1f} deg from the table normal "
          f"(depth agreement {s:.3f}, other axes fit as well up to "
          f"{ambig:.1f} deg away)")
  R = _rotation_a_to_b(cad.axial_axis, up)       # CAD up -> estimated axis
  T_cad_to_cam = np.eye(4)
  T_cad_to_cam[:3, :3] = R
  T_cad_to_cam[:3, 3] = center - R @ cad.base_center
  print(f"[+] Sphere fit: {n_in}/{len(pts)} inliers, center "
        f"{np.round(center*1000,1)} mm")
  return np.linalg.inv(T_cad_to_cam)


# ==============================================================================
# 5. ICP
# ==============================================================================
def icp_refine(scan, cad, T_scan_to_cad, params):
  """Multi-scale ICP, scan -> CAD, one estimator per stage ("plane" or
  "point"). Returns (T, fitness, rmse)."""
  reg = o3d.pipelines.registration
  T, fit, rmse = T_scan_to_cad, float("nan"), float("nan")
  for th, it, m in zip(params["icp_thresholds"], params["icp_iters"],
                       params["icp_methods"]):
    est = (reg.TransformationEstimationPointToPlane() if m == "plane"
           else reg.TransformationEstimationPointToPoint(False))
    r = reg.registration_icp(scan, cad.pcd, th, T, est,
                             reg.ICPConvergenceCriteria(max_iteration=it))
    T, fit, rmse = r.transformation, r.fitness, r.inlier_rmse
  return T, fit, rmse


def _score_c(obs, cad, T_cad_to_cam, params):
  Mp, Bp = project_silhouette(cad.dense_points, T_cad_to_cam, obs.K,
                              obs.mask.shape)
  return params["wB"] * box_iou(obs.box, Bp) + params["wM"] * mask_iou(obs.mask, Mp)


def table_yaw_sweep(obs, scan, cad, params, settings=CAMERA_SETTINGS):
  """Fallback global search for objects resting flat on the table.

  Up = table normal (depth ring around the mask), position = the mask
  centroid's viewing ray at half the object's height above the table, and
  yaw swept in steps. Each yaw is scored with C against the YOLO mask; the
  best few get the object's ICP. Uses the 2D evidence to pick the 3D
  solution, so log it separately from plain RANSAC in experiments.
  Returns T_scan_to_cad, or None if the table plane cannot be found.
  """
  ring_obs = obs
  n = _table_normal(ring_obs, params, settings)
  if n is None:
    return None
  # table plane offset from the ring (n . p + d = 0)
  r_in, r_out = params["table_ring_px"]
  m = obs.mask.astype(np.uint8)
  ring = (cv2.dilate(m, np.ones((2 * r_out + 1,) * 2, np.uint8)) > 0) & \
         ~(cv2.dilate(m, np.ones((2 * r_in + 1,) * 2, np.uint8)) > 0) & \
         (obs.depth > 0)
  vv, uu = np.nonzero(ring)
  z = obs.depth[vv, uu] / 1000.0
  ring_pts = np.stack([(uu - obs.K[0, 2]) * z / obs.K[0, 0],
                       (vv - obs.K[1, 2]) * z / obs.K[1, 1], z], 1)
  d = -float(np.median(ring_pts @ n))
  # object center: mask-centroid ray hits the plane lifted by half height
  height_axis = cad.axes[:, 2]
  proj = (cad.dense_points - cad.center) @ height_axis
  half_h = 0.5 * (proj.max() - proj.min())
  ys, xs = np.nonzero(obs.mask)
  ray = np.array([(xs.mean() - obs.K[0, 2]) / obs.K[0, 0],
                  (ys.mean() - obs.K[1, 2]) / obs.K[1, 1], 1.0])
  s = -(d - half_h) / (n @ ray)     # n.(s*ray) + d = half_h
  center_cam = s * ray
  R_up = _rotation_a_to_b(height_axis, n)

  cands = []
  for yaw in np.arange(0, 360, params["sweep_step_deg"]):
    R = _rot(n, yaw) @ R_up
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center_cam - R @ cad.center
    cands.append((_score_c(obs, cad, T, params), T))
  cands.sort(key=lambda c: c[0], reverse=True)
  best_C, best_T = -1.0, None
  for _, T in cands[:params["sweep_keep"]]:
    T_s2c, _, _ = icp_refine(scan, cad, np.linalg.inv(T), params)
    C = _score_c(obs, cad, np.linalg.inv(T_s2c), params)
    if C > best_C:
      best_C, best_T = C, T_s2c
  print(f"[+] Table yaw sweep: best C {best_C:.3f}")
  return best_T


# ==============================================================================
# 6. IOU CHECK (CAD projection, consistency, hypotheses, gate)
# ==============================================================================
def project_silhouette(points, T_cad_to_cam, K, img_shape, close_px=3):
  """Mproj and Bproj (Eq. 3-4) from a dense CAD splat plus closing."""
  h, w = img_shape[:2]
  p = (T_cad_to_cam[:3, :3] @ points.T).T + T_cad_to_cam[:3, 3]
  p = p[p[:, 2] > 1e-6]
  u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
  v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
  m = np.zeros((h, w), np.uint8)
  ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
  m[v[ok], u[ok]] = 1
  k = 2 * close_px + 1
  m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
  if not m.any():
    return m, [0, 0, 0, 0]
  # box from the unclipped projection, then clipped (not the points)
  box = [max(0, u.min()), max(0, v.min()), min(w - 1, u.max()), min(h - 1, v.max())]
  return m, [int(b) for b in box]


def box_iou(a, b):
  ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
  iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
  inter = ix * iy
  union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return inter / union if union > 0 else 0.0


def mask_iou(a, b):
  u = np.logical_or(a, b).sum()
  return float(np.logical_and(a, b).sum()) / u if u else 0.0


def symmetric_rotation_error_deg(R_gt, R_est, cad):
  """Eq. 18. For an axial object only the axis direction counts."""
  if cad.axial_axis is not None:
    a, b = R_gt @ cad.axial_axis, R_est @ cad.axial_axis
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
  best = 180.0
  for S in cad.symmetries:
    c = (np.trace((R_gt @ S[:3, :3]).T @ R_est) - 1) / 2
    best = min(best, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
  return best


def _same_pose(Ta, Tb, cad, rot_tol=5.0, trans_tol=0.005):
  """True if two CAD->camera poses are the same up to the object's symmetry."""
  if cad.axial_axis is not None:
    pa = Ta[:3, :3] @ cad.base_center + Ta[:3, 3]
    pb = Tb[:3, :3] @ cad.base_center + Tb[:3, 3]
    return (np.linalg.norm(pa - pb) < trans_tol and
            symmetric_rotation_error_deg(Ta[:3, :3], Tb[:3, :3], cad) < rot_tol)
  for S in cad.symmetries:
    Tbs = Tb @ S
    c = (np.trace(Ta[:3, :3].T @ Tbs[:3, :3]) - 1) / 2
    if (np.linalg.norm(Ta[:3, 3] - Tbs[:3, 3]) < trans_tol and
        np.degrees(np.arccos(np.clip(c, -1, 1))) < rot_tol):
      return True
  return False


def scan_fit_3d(scan, cad, T_cad_to_cam, dist=None):
  """F3: share of scan points within `dist` (default F3_DIST) of the CAD at
  this pose. Measured scan -> CAD, so CAD faces the camera cannot see do
  not count against the pose."""
  if scan is None or len(scan.points) == 0:
    return 0.0
  posed = copy.deepcopy(cad.pcd).transform(T_cad_to_cam)
  d = np.asarray(scan.compute_point_cloud_distance(posed))
  return float(np.mean(d < (F3_DIST if dist is None else dist)))


def iou_check(obs, scan, cad, T_scan_to_cad, params):
  """Scores the pose and its configured alternatives with C and gates it.

  Each hypothesis is (axis, degrees) about the CAD's principal axes
  ("long", "mid", "height") through the CAD center. Every candidate gets
  the object's ICP and is scored with C = wB*IB + wM*IM. The best one is
  kept; `margin` is its lead over the best candidate that is NOT the same
  pose up to symmetry.
  """
  axis_idx = {"long": 0, "mid": 1, "height": 2}
  cands = [("as registered", T_scan_to_cad)]
  for ax, deg in params["hypotheses"]:
    S = _rot_about(cad.center, cad.axes[:, axis_idx[ax]], deg)
    T_c2c = np.linalg.inv(T_scan_to_cad) @ S
    T_s2c, _, _ = icp_refine(scan, cad, np.linalg.inv(T_c2c), params)
    cands.append((f"{ax} {deg} deg", T_s2c))

  scored = []
  for name, T_s2c in cands:
    T_c2c = np.linalg.inv(T_s2c)
    Mp, Bp = project_silhouette(cad.dense_points, T_c2c, obs.K, obs.mask.shape)
    IB, IM = box_iou(obs.box, Bp), mask_iou(obs.mask, Mp)
    C = params["wB"] * IB + params["wM"] * IM
    scored.append(dict(name=name, T_cad_to_cam=T_c2c, IB=IB, IM=IM, C=C,
                       Mproj=Mp, Bproj=Bp))
  scored.sort(key=lambda s: s["C"], reverse=True)
  best = scored[0]
  rivals = [s for s in scored[1:]
            if not _same_pose(best["T_cad_to_cam"], s["T_cad_to_cam"], cad)]
  margin = best["C"] - rivals[0]["C"] if rivals else 1.0

  if best["C"] < params["tau_c"]:
    state, reason = "Conflict", "geometry"
  elif margin < params["delta"]:
    state, reason = "Conflict", "ambiguous orientation"
  else:
    state, reason = "Valid", ""
  ambig = obs.meta.get("axis_ambiguity_deg") if obs.meta else None
  if state == "Valid" and ambig is not None and \
      ambig > params.get("axis_ambig_deg", float("inf")):
    # the silhouette cannot see a tilted dome; the depth fit says the tilt
    # is not pinned down from this side, so look from another side
    state, reason = "Conflict", "axis ambiguous"
  F3 = scan_fit_3d(scan, cad, best["T_cad_to_cam"])
  if CHECK_3D and state == "Valid" and F3 < params.get("f3_min", 0.0):
    # outline agrees but the depth does not: shifted in depth or tilted
    state, reason = "Conflict", "3D check"
  UD = depth_incompleteness(obs)
  return dict(state=state, reason=reason, chosen=best["name"], F3=F3,
              T_cad_to_cam=best["T_cad_to_cam"], IB=best["IB"], IM=best["IM"],
              C=best["C"], margin=margin, UD=UD,
              U_live=0.7 * UD + 0.3 * (1 - best["C"]),
              Mproj=best["Mproj"], Bproj=best["Bproj"],
              axis_ambiguity_deg=ambig,
              all_scores=[(s["name"], round(s["C"], 3)) for s in scored])


# ==============================================================================
# DRIVER
# ==============================================================================
def estimate_pose(obs, cad, params):
  """Runs stages 2-6 for one observation. Returns the iou_check dict."""
  t0 = time.time()
  scan = build_point_cloud(obs, params)
  if len(scan.points) < 50:
    return dict(state="Conflict", reason="too few depth points",
                UD=depth_incompleteness(obs), C=0.0)
  if params["global_method"] == "fpfh_ransac":
    T = ransac_register(scan, cad, params, compute_fpfh(scan, params),
                        compute_fpfh(cad.pcd, params))
  else:
    T = ransac_register(scan, cad, params, obs=obs)
  if params["icp_thresholds"]:
    T, fit, rmse = icp_refine(scan, cad, T, params)
    print(f"[+] ICP fitness {fit:.3f}, rmse {rmse*1000:.2f} mm")
  out = iou_check(obs, scan, cad, T, params)
  out["used_fallback"] = False
  if out["state"] == "Conflict" and params.get("fallback") == "table_yaw_sweep":
    T_fb = table_yaw_sweep(obs, scan, cad, params)
    if T_fb is not None:
      out_fb = iou_check(obs, scan, cad, T_fb, params)
      if out_fb["C"] > out["C"]:
        out = out_fb
        out["used_fallback"] = True
  out["runtime_s"] = time.time() - t0
  out["scan"] = scan
  print(f"[+] {obs.object_key}: {out['state']} {out['reason']} "
        f"{'(fallback)' if out['used_fallback'] else ''}| "
        f"C={out['C']:.3f} (IB {out['IB']:.3f}, IM {out['IM']:.3f}) "
        f"margin={out['margin']:.3f} UD={out['UD']:.3f} "
        f"U_live={out['U_live']:.3f} F3={out.get('F3', float('nan')):.3f} | "
        f"{out['all_scores']}")
  return out


def estimate_pose_with_class_check(obs, cad_cache):
  """estimate_pose() for YOLO's class and for each class in CLASS_RIVALS.

  Every candidate CAD is registered to the same scan and scored with C
  against the same YOLO mask; the highest C wins (YOLO's class on a tie).
  The result records YOLO's class, the chosen class, every class's C and
  state, and whether the class was swapped. If the winner leads by less than
  CLASS_DELTA the state becomes Conflict "ambiguous class".
  """
  keys = [obs.object_key] + CLASS_RIVALS.get(obs.object_key, [])
  results = {}
  for k in keys:
    if k not in cad_cache:
      cad_cache[k] = load_cad_model(OBJECT_CONFIGS[k])
    if len(keys) > 1:
      print(f"[*] Class check: trying the {k} CAD")
    results[k] = estimate_pose(obs, cad_cache[k], OBJECT_CONFIGS[k])
  best = max(keys, key=lambda k: results[k].get("C", 0.0))
  out = results[best]
  out["yolo_object"] = obs.object_key
  out["final_object"] = best
  out["class_swapped"] = best != obs.object_key
  out["class_scores"] = {k: round(float(results[k].get("C", 0.0)), 3) for k in keys}
  out["class_states"] = {k: results[k]["state"] for k in keys}
  if len(keys) > 1:
    runner_up = max(results[k].get("C", 0.0) for k in keys if k != best)
    out["class_margin"] = float(out.get("C", 0.0) - runner_up)
    if out["state"] == "Valid" and out["class_margin"] < CLASS_DELTA:
      out["state"], out["reason"] = "Conflict", "ambiguous class"
    print(f"[+] Class check: YOLO said {obs.object_key}, chose {best}"
          f"{' (SWAPPED)' if out['class_swapped'] else ''} | "
          f"C {out['class_scores']} | margin {out['class_margin']:.3f} | "
          f"{out['state']} {out['reason']}")
  return out


def estimate_pose_same_view(obs, cad_cache, recapture, retries=None):
  """estimate_pose_with_class_check() with the same-view retry.

  On a Conflict (YOLO found the object) the camera stays put and
  recapture() takes up to `retries` fresh bursts; the first Valid one is
  used. recapture() returns an Observation, or None when YOLO finds
  nothing. Returns (obs, out); out["attempts"] counts the bursts at this
  view and out["next_step"] is "use pose" or "recovery move" (move the arm:
  best angle, then 90 deg counter-clockwise, then straight above).
  """
  retries = SAME_VIEW_RETRIES if retries is None else retries
  out, n = None, 0
  while True:
    if obs is None:
      out = dict(state="no detection", reason="no detection")
      break
    n += 1
    out = estimate_pose_with_class_check(obs, cad_cache)
    if out["state"] == "Valid" or n > retries:
      break
    print(f"[*] Same-view retry {n}/{retries}: {out['reason']}, new burst")
    obs = recapture()
  out["attempts"] = n
  out["next_step"] = "use pose" if out["state"] == "Valid" else "recovery move"
  if out["state"] != "Valid":
    print(f"[-] {out['state']} after {n} burst(s) at this view -> recovery move")
  return obs, out


def show_3d_overlap(scan, cad, T_cad_to_cam, title="CAD vs scan"):
  """Posed CAD (cyan) and the captured cloud (yellow), both in the camera
  frame, viewed from the camera. Also prints how well they overlap: the
  share of scan points within 3 / 5 mm of the CAD and the median distance.
  Close the window to continue to the IoU view.
  """
  cad_posed = copy.deepcopy(cad.pcd)
  cad_posed.transform(T_cad_to_cam)
  cad_posed.paint_uniform_color([0, 0.651, 0.929])
  scan_vis = copy.deepcopy(scan)
  scan_vis.paint_uniform_color([1, 0.706, 0])

  d = np.asarray(scan.compute_point_cloud_distance(cad_posed))
  if len(d):
    print(f"[+] 3D overlap: {np.mean(d < 0.003)*100:.1f}% of scan points "
          f"within 3 mm, {np.mean(d < 0.005)*100:.1f}% within 5 mm, "
          f"median {np.median(d)*1000:.2f} mm")

  frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
  frame.transform(T_cad_to_cam)  # CAD axes at the estimated pose
  o3d.visualization.draw_geometries(
      [cad_posed, scan_vis, frame], window_name=title,
      lookat=scan.get_center(), front=[0, 0, -1], up=[0, -1, 0], zoom=0.6)


def visualize_result(obs, out):
  img = obs.color.copy()
  cv2.drawContours(img, cv2.findContours(obs.mask.astype(np.uint8),
                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                   (0, 255, 0), 1)
  cv2.drawContours(img, cv2.findContours(out["Mproj"].astype(np.uint8),
                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                   (0, 0, 255), 1)
  x1, y1, x2, y2 = obs.box
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
  px1, py1, px2, py2 = out["Bproj"]
  cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 1)
  color = (0, 200, 0) if out["state"] == "Valid" else (0, 0, 255)
  name = out.get("final_object", obs.object_key)
  if out.get("class_swapped"):
    name += f" (YOLO: {obs.object_key})"
  cv2.putText(img, f"{name}: {out['state']} C={out['C']:.2f} "
              f"m={out['margin']:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
              0.6, color, 2)
  cv2.imshow("V-GAP check (green = YOLO, red = CAD)", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def save_trial(obs, out, root="data/raw"):
  """Saves the plan's per-trial record (RGB, depth, mask, pose, metrics)."""
  d = os.path.join(root, f"{obs.object_key}_{obs.meta.get('timestamp', 'na')}")
  os.makedirs(d, exist_ok=True)
  cv2.imwrite(os.path.join(d, "rgb.png"), obs.color)
  cv2.imwrite(os.path.join(d, "depth_mm.png"), obs.depth)
  cv2.imwrite(os.path.join(d, "mask.png"), obs.mask.astype(np.uint8) * 255)
  meta = dict(object=obs.object_key, class_name=obs.class_name,
              confidence=obs.confidence, box=obs.box, K=obs.K.tolist(),
              settings=obs.meta.get("settings"),
              **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in out.items() if k not in ("Mproj", "scan")})
  with open(os.path.join(d, "result.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)
  print(f"[+] Trial saved to {d}")


def main():
  from ultralytics import YOLO
  yolo_model = YOLO(YOLO_MODEL_PATH)
  print(f"[*] YOLO classes: {yolo_model.names}")
  pipeline, align, K = start_realsense()
  cad_cache = {}
  try:
    while True:
      obs = capture_observation(pipeline, align, K, yolo_model)
      if obs is None:
        break
      obs, out = estimate_pose_same_view(
          obs, cad_cache,
          lambda: capture_observation_now(pipeline, align, K, yolo_model))
      final = out.get("final_object")
      if obs is not None and "T_cad_to_cam" in out:
        print("[+] T_cad_to_cam:\n", np.round(out["T_cad_to_cam"], 5))
        save_trial(obs, out)
        show_3d_overlap(out["scan"], cad_cache[final],
                        out["T_cad_to_cam"],
                        f"{final}: CAD (cyan) vs scan (yellow)")
        visualize_result(obs, out)
  finally:
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
