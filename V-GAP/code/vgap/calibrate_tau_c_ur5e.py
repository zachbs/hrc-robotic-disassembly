"""Collects labelled C values for tau_c calibration: UR5e view sweep + V-GAP.

For one object (and one filament color) per run, the wrist camera visits
tilts 0/15/30/45/60 deg from the top normal at several azimuths around the
object. At every view it runs the normal pipeline
(vgap_pipeline.estimate_pose_with_class_check) and logs C, the gate state
and whether the pose is actually correct. Then recommend_tau_c.py turns the
CSVs into one tau_c per object.

How "correct" is decided (--label):
  reference  (default) At the first view you check the pose by eye (3D
             overlap + silhouette window) and accept it with 'y'. That pose,
             moved into the robot base frame with the TCP pose and the
             hand-eye matrix, is the ground truth for every other view:
             correct <= 5 deg & <= 5 mm, wrong > 10 deg or > 10 mm (or the
             wrong class), in between = borderline. The object must not move
             during the sweep.
  manual     No hand-eye needed: you judge every view yourself
             (y = correct, n = wrong, b = borderline, s = skip).

Injected errors (on by default, --no-inject to switch off): at every view
whose pose is correct, the true pose is deliberately disturbed (yaw 15/30/
90/180 deg, a flip, 12/20 mm shifts, plus tilt and depth shift) and C is
computed for each wrong pose, raw and after the object's ICP. ICP slides
most of them back to the right pose; the ones it leaves wrong are the local
minima the real pipeline can return, and those set tau_c. Your pipeline
rarely produces wrong poses on its own, and without wrong examples there is
nothing to put tau_c between.

Robot (--robot):
  dry-run  (default) No robot. Prints every TCP target as UR [x,y,z,rx,ry,rz]
           and, with the real camera, waits for you to jog there and press
           Enter. Good first run: check the poses on the pendant.
  ur-rtde  Moves a real UR5e through ur_rtde (pip install ur_rtde). UNTESTED
           ON HARDWARE: slow moveL (5 cm/s), every move needs Enter unless
           --no-confirm, and targets are refused if the camera would get
           closer than --min-dist to the object or the TCP below --min-z.
           Keep a hand on the e-stop for the first run.
  sim      Simulated robot + simulated RealSense (sim_stress_test.py
           renders, simulated YOLO mask). For testing this script offline.

Hand-eye: --hand-eye T_flange_cam.npy (4x4, camera pose in the TCP frame,
meters, so T_base_cam = T_base_tcp @ T; same convention as your
loop_pickup code) or a .json holding {"T_tool_cam": 4x4}. Keep the pendant's
active TCP the same as when the matrix was made (the flange).
A hand-measured matrix is usually off by a few cm, which would mislabel
views (an offset error e shows up as up to ~e of pose error between views).
So by default the sweep also SOLVES the hand-eye from its own views: the
object is static, so robot pose x hand-eye x object-in-camera must give the
same object pose in the base frame from every view (AX = XB, Park-Martin
method, RANSAC over views, anchored on the pose you approved). The refined
matrix is saved as handeye_refined_<object>_<time>.npy and used for the
labels. Solve it once with the L or T block (not the hemisphere: its yaw is
unobservable), then pass that file with --no-refine-hand-eye for the rest.

Examples (from the project root):
  # print the planned TCP poses, no robot, no camera
  python code/vgap/calibrate_tau_c_ur5e.py --object l_block --color white \
      --target 0.30,-0.45,0.013 --hand-eye T_flange_cam.npy --plan-only
  # real sweep, robot moves, you confirm the first pose
  python code/vgap/calibrate_tau_c_ur5e.py --object l_block --color white \
      --robot ur-rtde --robot-ip 192.168.1.5 --hand-eye T_flange_cam.npy \
      --serial 337122075128
  # offline test
  python code/vgap/calibrate_tau_c_ur5e.py --object t_block --color grey --robot sim
Output: data/calib/calib_<object>_<color>_<time>.csv (+ per-view trial
folders under data/calib/trials when the real camera is used).
"""
import argparse
import contextlib
import copy
import csv
import io
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vgap_pipeline as vp  # noqa: E402

GOOD_ROT, GOOD_TRANS = 5.0, 5.0      # deg, mm (same as the simulations)
BAD_ROT, BAD_TRANS = 10.0, 10.0
# (name, kind, axis or direction, amount, group). Rotations are about the
# CAD's principal axes through its center, shifts are in the camera frame.
# group "in_plane": changes the silhouette, so C should notice it.
# group "out_of_plane": tilt / depth shift; the 2D silhouette barely changes,
# so C cannot catch these (a 2D gate's blind spot; reported, not used for tau_c).
# Every one is also run through the object's ICP: whatever wrong pose ICP
# settles in is the kind of local minimum the real pipeline can return.
INJECTIONS = [
    ("yaw +15", "rot", "height", 15, "in_plane"),
    ("yaw -15", "rot", "height", -15, "in_plane"),
    ("yaw +30", "rot", "height", 30, "in_plane"),
    ("yaw 90", "rot", "height", 90, "in_plane"),
    ("yaw 180", "rot", "height", 180, "in_plane"),
    ("flip 180", "rot", "long", 180, "in_plane"),
    ("shift x 12mm", "shift", (1, 0, 0), 0.012, "in_plane"),
    ("shift y 20mm", "shift", (0, 1, 0), 0.020, "in_plane"),
    ("tilt 15", "rot", "long", 15, "out_of_plane"),
    ("shift z 15mm", "shift", (0, 0, 1), 0.015, "out_of_plane"),
]
FIELDS = ["object", "color", "view", "tilt_deg", "azimuth_deg", "rep", "sample",
          "perturbation", "yolo_object", "final_object", "C", "IB", "IM",
          "margin", "class_margin", "axis_ambiguity_deg", "state", "reason",
          "used_fallback", "chosen", "UD", "rot_err_deg", "trans_err_mm",
          "inject_kind", "inject_group", "label", "label_source",
          "caught_by_other_check", "true_rot_err_deg",
          "true_trans_err_mm", "tcp_pose", "T_cad_to_cam", "trial_dir",
          "timestamp"]


# ==============================================================================
# Geometry helpers
# ==============================================================================
def look_at(target, tilt_deg, az_deg, dist):
  """Camera pose in the base frame (T_base_cam, camera z toward the target).

  Base z must point up (UR5e base frame, table horizontal). Tilt is measured
  from straight down; azimuth turns around the target. Same convention as
  sim_stress_test.camera_pose().
  """
  t, a = np.radians(tilt_deg), np.radians(az_deg)
  cpos = target + dist * np.array([np.sin(t) * np.cos(a),
                                   np.sin(t) * np.sin(a), np.cos(t)])
  z = (target - cpos) / np.linalg.norm(target - cpos)
  ref = np.array([np.cos(a + np.pi / 2), np.sin(a + np.pi / 2), 0.0])
  x = np.cross(z, ref) if tilt_deg >= 1 else ref
  x = x - (x @ z) * z
  x /= np.linalg.norm(x)
  T = np.eye(4)
  T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
  T[:3, 3] = cpos
  return T


def to_ur_pose(T):
  rvec, _ = cv2.Rodrigues(T[:3, :3])
  return [float(v) for v in T[:3, 3]] + [float(v) for v in rvec.ravel()]


def from_ur_pose(p):
  T = np.eye(4)
  T[:3, :3], _ = cv2.Rodrigues(np.asarray(p[3:], float))
  T[:3, 3] = p[:3]
  return T


def pose_error(T_ref, T, cad):
  """Symmetry-aware rotation (deg) and translation (mm) error, as in the sims."""
  e_r = vp.symmetric_rotation_error_deg(T_ref[:3, :3], T[:3, :3], cad)
  if cad.axial_axis is not None:
    e_t = np.linalg.norm((T_ref[:3, :3] - T[:3, :3]) @ cad.base_center +
                         T_ref[:3, 3] - T[:3, 3]) * 1000
  else:
    e_t = min(np.linalg.norm((T_ref @ S)[:3, 3] - T[:3, 3])
              for S in cad.symmetries) * 1000
  return float(e_r), float(e_t)


def label_of(e_r, e_t, wrong_class=False):
  if wrong_class:
    return "wrong"
  if e_r <= GOOD_ROT and e_t <= GOOD_TRANS:
    return "correct"
  if e_r > BAD_ROT or e_t > BAD_TRANS:
    return "wrong"
  return "borderline"


def plan_views(tilts, azimuths, reps):
  """Ordered (tilt, azimuth, rep) list; azimuth sweeps alternate direction so
  consecutive views stay close and the wrist never wraps more than once."""
  views = []
  for i, t in enumerate(tilts):
    az = azimuths if i % 2 == 0 else azimuths[::-1]
    for a in az:
      for r in range(reps):
        views.append((t, a, r))
  return views


# ==============================================================================
# Robot adapters  (the ONLY place that moves hardware)
# ==============================================================================
class DryRunRobot:
  """Moves nothing. Prints targets; optionally waits for you to jog there."""
  moves_hardware = False

  def __init__(self, wait_for_operator=False, start_pose=None):
    self.T = np.eye(4) if start_pose is None else start_pose
    self.wait = wait_for_operator

  def get_tcp(self):
    return self.T.copy()

  def move_tcp(self, T_base_tool, label=""):
    print(f"[dry-run] {label} TCP target (UR pose, m / rotvec): "
          f"{np.round(to_ur_pose(T_base_tool), 4).tolist()}")
    if self.wait:
      input("          jog the robot there, then press Enter ")
    self.T = T_base_tool.copy()

  def close(self):
    pass


class URRtdeRobot:
  """UR5e through ur_rtde.  UNTESTED ON HARDWARE.

  Every move is a slow moveL. Before each move it checks the target against
  the safety limits and (unless confirm=False) waits for Enter.
  """
  moves_hardware = True

  def __init__(self, ip, speed=0.05, acc=0.10, confirm=True):
    import rtde_control
    import rtde_receive
    self.c = rtde_control.RTDEControlInterface(ip)
    self.r = rtde_receive.RTDEReceiveInterface(ip)
    self.speed, self.acc, self.confirm = speed, acc, confirm

  def get_tcp(self):
    return from_ur_pose(self.r.getActualTCPPose())

  def move_tcp(self, T_base_tool, label=""):
    p = to_ur_pose(T_base_tool)
    print(f"[robot] {label} moveL to {np.round(p, 4).tolist()}")
    if self.confirm:
      if input("        Enter = move, q = stop ").strip().lower() == "q":
        raise KeyboardInterrupt("stopped by operator")
    if not self.c.moveL(p, self.speed, self.acc):
      raise RuntimeError("moveL failed (protective stop / unreachable?)")

  def close(self):
    with contextlib.suppress(Exception):
      self.c.stopScript()


def check_target(T_base_cam, T_base_tool, target, min_dist, min_z, max_reach,
                 tool_tip=None, tip_clear=0.10):
  d = np.linalg.norm(T_base_cam[:3, 3] - target)
  if d < min_dist:
    raise ValueError(f"camera would be {d*1000:.0f} mm from the object (< min-dist)")
  if tool_tip is not None:   # gripper fingertips, given in the TCP frame
    tip = T_base_tool[:3, :3] @ tool_tip + T_base_tool[:3, 3]
    if np.linalg.norm(tip - target) < tip_clear:
      raise ValueError(f"gripper tip would be {np.linalg.norm(tip - target)*1000:.0f}"
                       " mm from the object")
    if tip[2] < target[2]:
      raise ValueError("gripper tip would go below the object center height")
  if T_base_tool[2, 3] < min_z:
    raise ValueError(f"TCP z {T_base_tool[2, 3]:.3f} m is below --min-z {min_z}")
  if np.linalg.norm(T_base_tool[:2, 3]) > max_reach:
    raise ValueError("target outside the UR5e reach limit set by --max-reach")


# ==============================================================================
# Camera adapters
# ==============================================================================
class RealSenseCamera:
  """Hands-free version of vp.capture_observation(): waits for the arm to
  settle, takes the best known detection over a few frames, then the burst."""
  simulated = False

  def __init__(self, serial=None, settle_s=1.0, yolo_tries=15):
    from ultralytics import YOLO
    self.yolo = YOLO(vp.YOLO_MODEL_PATH)
    self.settings = s = vp.CAMERA_SETTINGS
    if serial is None:
      self.pipeline, self.align, self.K = vp.start_realsense()
    else:   # same as vp.start_realsense(), but picks the wrist camera
      import pyrealsense2 as rs
      self.pipeline = rs.pipeline()
      cfg = rs.config()
      cfg.enable_device(serial)
      cfg.enable_stream(rs.stream.color, s["width"], s["height"], rs.format.bgr8, s["fps"])
      cfg.enable_stream(rs.stream.depth, s["width"], s["height"], rs.format.z16, s["fps"])
      prof = self.pipeline.start(cfg)
      self.align = rs.align(rs.stream.color)
      it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
      self.K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.0]])
      ds = prof.get_device().first_depth_sensor()
      if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, s["laser_power"])
      if ds.supports(rs.option.enable_auto_exposure):
        ds.set_option(rs.option.enable_auto_exposure, False)
      if ds.supports(rs.option.exposure):
        ds.set_option(rs.option.exposure, s["exposure_us"])
      time.sleep(1)
      print(f"[+] RealSense {serial} started")
    self.settle_s, self.tries = settle_s, yolo_tries
    # detection filters (vp.select_detection): the gate eval sets target
    # (base frame) and X (T_flange_cam) once the object position is known
    self.target, self.X = None, None
    self.gripper_roi, self.max_dist = "default", None
    self.last_filter = {}

  def _frames(self):
    while True:
      f = self.align.process(self.pipeline.wait_for_frames())
      d, c = f.get_depth_frame(), f.get_color_frame()
      if d and c:
        return np.asanyarray(d.get_data()).copy(), np.asanyarray(c.get_data()).copy()

  def capture(self, T_base_tool=None):
    t0 = time.time()
    while time.time() - t0 < self.settle_s:   # flush frames taken while moving
      self._frames()
    T_bc = None
    if T_base_tool is not None and self.X is not None:
      T_bc = T_base_tool @ self.X
    for _ in range(self.tries):
      depth, color = self._frames()
      res = self.yolo(color, conf=self.settings["yolo_conf"], agnostic_nms=True,
                      verbose=False)[0]
      idx, self.last_filter = vp.select_detection(
          res, depth.shape, depth, self.K, T_bc, self.target,
          self.gripper_roi, max_dist=self.max_dist)
      if idx is not None and res.masks is not None:
        break
    else:
      return None
    burst = [depth]
    while len(burst) < self.settings["frames_to_capture"]:
      burst.append(self._frames()[0])
    h, w = depth.shape
    mask = cv2.resize(res.masks.data[idx].cpu().numpy(), (w, h),
                      interpolation=cv2.INTER_NEAREST) > 0.5
    median = vp.temporal_median_ignore_zeros(burst)
    filt = cv2.bilateralFilter(median.astype(np.float32), d=5,
                               sigmaColor=15.0, sigmaSpace=5.0)
    filt[median == 0] = 0
    name = res.names[int(res.boxes.cls[idx])]
    return vp.Observation(
        color=color, depth=filt.astype(np.uint16), mask=mask,
        box=[int(v) for v in res.boxes.xyxy[idx].cpu().numpy()],
        class_name=name, object_key=vp.object_key(name),
        confidence=float(res.boxes.conf[idx]), K=self.K,
        meta=dict(settings=self.settings, det_filter=self.last_filter,
                  timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")))

  def close(self):
    self.pipeline.stop()


class SimCamera:
  """Renders the wrist view with sim_stress_test.py (no YOLO: simulated mask,
  YOLO's class assumed right). Knows the true pose, for testing only."""
  simulated = True

  def __init__(self, obj, color, seed=0, T_tool_cam_true=None):
    import sim_stress_test as st
    import open3d as o3d
    self.st, self.obj, self.color = st, obj, color
    self.rng = np.random.default_rng(seed)
    mesh = o3d.io.read_triangle_mesh(os.path.join(st.ROOT, st.MESHES[obj]))
    mesh.scale(0.001, center=np.zeros(3))
    self.mesh = mesh
    T_ow, self.props, _, _ = st.place_object(mesh, "flat", self.rng)
    # put the object at a typical spot in front of the robot
    T_ow[:3, 3] += np.array([0.30, -0.45, 0.0])   # near your OBSERVE_POSE
    self.T_base_obj = T_ow
    self.K = st.K
    self.X = T_tool_cam_true      # the real mount, unknown to the script

  def object_center(self):
    return np.asarray(copy.deepcopy(self.mesh).transform(self.T_base_obj)
                      .vertices).mean(0)

  def capture(self, T_base_tool):
    T_cw = np.linalg.inv(T_base_tool @ self.X)
    bgr, depth, true_mask = self.st.render(self.mesh, self.T_base_obj, self.props,
                                           T_cw, self.color, self.rng)
    if true_mask.sum() < 300:
      return None
    mask = self.st.sim_yolo_mask(true_mask, self.rng)
    return vp.Observation(color=bgr, depth=depth, mask=mask,
                          box=self.st.box_of(mask), class_name=self.obj,
                          object_key=self.obj, confidence=1.0, K=self.K,
                          meta=dict(timestamp=datetime.now().strftime("%H%M%S_%f")))

  def true_pose_cam(self, T_base_tool):
    return np.linalg.inv(T_base_tool @ self.X) @ self.T_base_obj

  def close(self):
    pass


# ==============================================================================
# Per-view work
# ==============================================================================
def run_pipeline(obs, cad_cache, quiet):
  out_buf = io.StringIO()
  with contextlib.redirect_stdout(out_buf) if quiet else contextlib.nullcontext():
    out = vp.estimate_pose_with_class_check(obs, cad_cache)
  return out


def caught_by_other_check(out, params):
  """True if the margin, class or axis check would reject this pose anyway."""
  if out.get("margin", 1.0) < params["delta"]:
    return True
  if out.get("class_margin", 1.0) < vp.CLASS_DELTA:
    return True
  amb = out.get("axis_ambiguity_deg")
  return amb is not None and amb > params.get("axis_ambig_deg", float("inf"))


def score_pose(obs, cad, T_c2c, params):
  Mp, Bp = vp.project_silhouette(cad.dense_points, T_c2c, obs.K, obs.mask.shape)
  IB, IM = vp.box_iou(obs.box, Bp), vp.mask_iou(obs.mask, Mp)
  return IB, IM, params["wB"] * IB + params["wM"] * IM


def injected_rows(obs, scan, cad, params, T_true_cam):
  """C of deliberately wrong poses around the true one, raw and after ICP.

  An ICP result that did not move (no correspondences within its threshold)
  is dropped: it is the raw pose again, not something ICP would return.
  """
  rows = []
  ax = {"long": 0, "mid": 1, "height": 2}
  for name, kind, what, amt, group in INJECTIONS:
    if kind == "rot":
      if cad.axial_axis is not None and what == "height":
        continue  # yaw of the hemisphere is unobservable
      T = T_true_cam @ vp._rot_about(cad.center, cad.axes[:, ax[what]], amt)
    else:
      T = T_true_cam.copy()
      T[:3, 3] += amt * np.asarray(what, float)
    variants = [("raw", T)]
    if params["icp_thresholds"] and scan is not None and len(scan.points) >= 50:
      T_s2c, _, _ = vp.icp_refine(scan, cad, np.linalg.inv(T), params)
      Ti = np.linalg.inv(T_s2c)
      if pose_error(T, Ti, cad)[1] > 0.1 or \
          vp.symmetric_rotation_error_deg(T[:3, :3], Ti[:3, :3], cad) > 0.05:
        variants.append(("icp", Ti))
    for tag, Tv in variants:
      IB, IM, C = score_pose(obs, cad, Tv, params)
      e_r, e_t = pose_error(T_true_cam, Tv, cad)
      rows.append(dict(sample="injected", perturbation=f"{name} {tag}",
                       inject_kind=tag, inject_group=group,
                       C=round(C, 4), IB=round(IB, 4), IM=round(IM, 4),
                       rot_err_deg=round(e_r, 2), trans_err_mm=round(e_t, 2),
                       label=label_of(e_r, e_t), label_source="injected",
                       caught_by_other_check=False))
  return rows


def _rot_deg(Ra, Rb):
  return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def park_martin(T_bts, T_ccs):
  """Closed-form AX = XB (Park & Martin 1994), no OpenCV contrib needed.

  A = relative TCP motion, B = relative object motion seen by the camera,
  over all view pairs. Returns the 4x4 camera pose in the TCP frame or None.
  """
  As, Bs = [], []
  for i in range(len(T_bts)):
    for j in range(i + 1, len(T_bts)):
      A = np.linalg.inv(T_bts[j]) @ T_bts[i]
      ang = _rot_deg(A[:3, :3], np.eye(3))
      if ang < 5 or ang > 150:   # too small to inform / log() unstable near 180
        continue
      As.append(A)
      Bs.append(T_ccs[j] @ np.linalg.inv(T_ccs[i]))
  if len(As) < 2:
    return None
  M = np.zeros((3, 3))
  for A, B in zip(As, Bs):
    a = cv2.Rodrigues(A[:3, :3])[0].ravel()
    b = cv2.Rodrigues(B[:3, :3])[0].ravel()
    M += np.outer(b, a)
  w, V = np.linalg.eigh(M.T @ M)
  if w.min() < 1e-9:
    return None                     # rotations too similar (degenerate)
  R = V @ np.diag(w ** -0.5) @ V.T @ M.T
  C = np.vstack([A[:3, :3] - np.eye(3) for A in As])
  d = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
  t = np.linalg.lstsq(C, d, rcond=None)[0]
  X = np.eye(4)
  X[:3, :3], X[:3, 3] = R, t
  return X if np.all(np.isfinite(X)) else None


def solve_hand_eye(T_bts, T_ccs, cad, X0, ref, iters=400, tol_mm=5.0,
                   tol_deg=3.0, seed=0):
  """Hand-eye (AX = XB) from views of one static object.

  T_bts: TCP poses (base frame); T_ccs: pipeline poses (CAD -> camera);
  ref: index of the view you approved. The object pose in the base frame,
  T_bt @ X @ T_cc, must be the same from every view. Symmetric objects are
  first snapped to the symmetry copy closest in rotation to the reference
  (rotation uses X0's rotation, which a hand measurement gets roughly
  right). RANSAC over 4-view subsets drops views where the pipeline was
  wrong. Returns (X, inlier mask, per-view residuals (mm, deg), object
  pose in the base frame averaged over the inlier views) or None.
  """
  if cad.axial_axis is not None:
    return None
  n = len(T_bts)
  P_ref = T_bts[ref] @ X0 @ T_ccs[ref]
  T_ccs = list(T_ccs)
  for i in range(n):
    T_ccs[i] = min((T_ccs[i] @ S for S in cad.symmetries),
                   key=lambda T: _rot_deg(P_ref[:3, :3], (T_bts[i] @ X0 @ T)[:3, :3]))

  def fit(idx):
    return park_martin([T_bts[i] for i in idx], [T_ccs[i] for i in idx])

  def residuals(X):
    P = [T_bts[i] @ X @ T_ccs[i] for i in range(n)]
    c = np.median([p[:3, 3] for p in P], axis=0)   # robust object position
    Rr = P[ref][:3, :3]
    return np.array([[np.linalg.norm(p[:3, 3] - c) * 1000,
                      _rot_deg(p[:3, :3], Rr)] for p in P])

  rng = np.random.default_rng(seed)
  best = None
  others = [i for i in range(n) if i != ref]
  if len(others) < 4:
    return None
  for _ in range(iters):
    idx = [ref] + list(rng.choice(others, 3, replace=False))
    X = fit(idx)
    if X is None:
      continue
    r = residuals(X)
    inl = (r[:, 0] < tol_mm) & (r[:, 1] < tol_deg)
    score = (inl.sum(), -r[inl].sum())
    if best is None or score > best[0]:
      best = (score, inl)
  if best is None or best[1].sum() < 5:
    return None
  inl = best[1]
  for _ in range(3):   # refit on the inliers, then re-check
    X = fit(np.nonzero(inl)[0])
    if X is None:
      return None
    r = residuals(X)
    new = (r[:, 0] < tol_mm) & (r[:, 1] < tol_deg)
    if new.sum() < 5 or (new == inl).all():
      break
    inl = new
  P = [T_bts[i] @ X @ T_ccs[i] for i in np.nonzero(inl)[0]]
  U, _, Vt = np.linalg.svd(sum(p[:3, :3] for p in P))   # chordal rotation mean
  P_mean = np.eye(4)
  P_mean[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
  P_mean[:3, 3] = np.mean([p[:3, 3] for p in P], axis=0)
  return X, inl, r, P_mean


def show_and_ask(obs, out, cad, gui):
  """Shows the result and asks the operator to judge it. Returns y/n/b/s."""
  if gui and "T_cad_to_cam" in out:
    vp.show_3d_overlap(out["scan"], cad, out["T_cad_to_cam"],
                       "CAD (cyan) vs scan (yellow): close to continue")
    vp.visualize_result(obs, out)
  while True:
    k = input("  Pose correct? y = yes (<= 5 deg / 5 mm), n = no, "
              "b = borderline, s = skip: ").strip().lower()
    if k in ("y", "n", "b", "s"):
      return k


# ==============================================================================
# Main
# ==============================================================================
def parse_list(s, typ=float):
  return [typ(v) for v in s.split(",") if v.strip()]


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--object", required=True, choices=list(vp.OBJECT_CONFIGS))
  ap.add_argument("--color", required=True, help="white / grey / dark_reflective")
  ap.add_argument("--robot", default="dry-run", choices=["dry-run", "ur-rtde", "sim"])
  ap.add_argument("--robot-ip", default=None)
  ap.add_argument("--hand-eye", default=None,
                  help='T_flange_cam.npy (4x4) or json with {"T_tool_cam": 4x4}')
  ap.add_argument("--no-refine-hand-eye", action="store_true",
                  help="use --hand-eye as is (e.g. a matrix refined by an earlier sweep)")
  ap.add_argument("--serial", default=None,
                  help="RealSense serial of the wrist camera (needed with 2 cameras)")
  ap.add_argument("--tool-tip", default="0,0,0.20",
                  help="gripper fingertip in the TCP frame, m ('none' = no check)")
  ap.add_argument("--target", default=None,
                  help="object center in the base frame, x,y,z in m. If left out, "
                       "it is measured from a first capture at the current pose")
  ap.add_argument("--tilts", default="0,15,30,45,60")
  ap.add_argument("--azimuths", default="0,90,180,270")
  ap.add_argument("--dist", type=float, default=0.40, help="camera to object, m")
  ap.add_argument("--reps", type=int, default=1, help="captures per view")
  ap.add_argument("--label", default="reference", choices=["reference", "manual"])
  ap.add_argument("--ref-view", default="45,0", help="tilt,azimuth of the view used "
                  "for the reference pose (the most reliable sim view is 45-60)")
  ap.add_argument("--no-inject", action="store_true")
  ap.add_argument("--no-gui", action="store_true", help="no OpenCV/Open3D windows")
  ap.add_argument("--no-confirm", action="store_true",
                  help="ur-rtde: do not ask before every move")
  ap.add_argument("--min-dist", type=float, default=0.25)
  ap.add_argument("--min-z", type=float, default=0.05, help="lowest TCP z, m")
  ap.add_argument("--max-reach", type=float, default=0.85,
                  help="max TCP distance from the base axis, m (UR5e reach 0.85)")
  ap.add_argument("--plan-only", action="store_true",
                  help="print the TCP poses and exit")
  ap.add_argument("--assume-yes", action="store_true",
                  help="sim only: accept the reference pose without asking")
  ap.add_argument("--seed", type=int, default=0, help="sim only")
  ap.add_argument("--out", default="data/calib")
  a = ap.parse_args()

  params = vp.OBJECT_CONFIGS[a.object]
  tilts, azs = parse_list(a.tilts), parse_list(a.azimuths)
  views = plan_views(tilts, azs, a.reps)
  quiet = a.robot == "sim"
  gui = not (a.no_gui or a.robot == "sim")

  # hand-eye
  X_sim_true = np.eye(4)                       # sim only: the "real" mount
  X_sim_true[:3, 3] = [0.0, 0.065, 0.035]
  X_sim_true[:3, :3] = vp._rot([0, 0, 1], 3.0)
  if a.hand_eye:
    if a.hand_eye.endswith(".npy"):
      T_tool_cam = np.load(a.hand_eye).astype(float)
    else:
      with open(a.hand_eye) as f:
        T_tool_cam = np.asarray(json.load(f)["T_tool_cam"], float)
    assert T_tool_cam.shape == (4, 4), "hand-eye must be a 4x4 matrix"
  elif a.robot == "sim":
    T_tool_cam = X_sim_true.copy()
  else:
    T_tool_cam = None
  tool_tip = None if a.tool_tip.lower() == "none" else np.asarray(parse_list(a.tool_tip))
  if T_tool_cam is None and (a.label == "reference" or a.robot == "ur-rtde"):
    raise SystemExit("--hand-eye is needed for --label reference and for moving "
                     "the robot. Without it use --robot dry-run --label manual "
                     "and jog to each view yourself.")
  if a.assume_yes and a.robot != "sim":
    raise SystemExit("--assume-yes is only allowed with --robot sim.")
  T_cam_tool = np.linalg.inv(T_tool_cam) if T_tool_cam is not None else None

  if a.plan_only:
    if a.target is None or T_tool_cam is None:
      raise SystemExit("--plan-only needs --target and --hand-eye")
    tgt = np.asarray(parse_list(a.target))
    for t, az, r in views:
      Tc = look_at(tgt, t, az, a.dist)
      Tt = Tc @ T_cam_tool
      try:
        check_target(Tc, Tt, tgt, a.min_dist, a.min_z, a.max_reach, tool_tip)
      except ValueError as e:
        print(f"tilt {t:4.0f} az {az:5.0f} rep {r}: SKIPPED ({e})")
        continue
      print(f"tilt {t:4.0f} az {az:5.0f} rep {r}: "
            f"{np.round(to_ur_pose(Tt), 4).tolist()}")
    return

  # adapters
  if a.robot == "sim":
    import sim_stress_test as st
    for k, rel in st.MESHES.items():   # same CADs the simulation renders
      vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(st.ROOT, rel)
      vp.OBJECT_CONFIGS[k]["cad_units"] = "mm"
    cam = SimCamera(a.object, a.color, a.seed, X_sim_true)
    if a.target is None:
      a.target = ",".join(str(v) for v in cam.object_center())
    start = look_at(np.asarray(parse_list(a.target)), 0, 0, a.dist) @ T_cam_tool
    robot = DryRunRobot(False, start)
  else:
    cam = RealSenseCamera(a.serial)
    if a.robot == "ur-rtde":
      if not a.robot_ip:
        raise SystemExit("--robot-ip is required for ur-rtde")
      robot = URRtdeRobot(a.robot_ip, confirm=not a.no_confirm)
    else:
      robot = DryRunRobot(wait_for_operator=True)
      if a.label == "reference":
        print("[!] dry-run + reference labels: the script cannot read the TCP, so "
              "every pose must be jogged to exactly as printed.")

  cad_cache = {}
  for k in [a.object] + vp.CLASS_RIVALS.get(a.object, []):
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
      cad_cache[k] = vp.load_cad_model(vp.OBJECT_CONFIGS[k])
  cad = cad_cache[a.object]

  os.makedirs(a.out, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(a.out, f"calib_{a.object}_{a.color}_{stamp}.csv")
  fcsv = open(csv_path, "w", newline="")
  writer = csv.DictWriter(fcsv, fieldnames=FIELDS, extrasaction="ignore")
  writer.writeheader()

  def tcp_cam():
    T = robot.get_tcp()
    return T @ T_tool_cam if T_tool_cam is not None else None

  try:
    # ---------------------------------------------------------------- target
    if a.target is None:
      print("[*] No --target: capturing at the current pose to find the object.")
      obs = cam.capture(robot.get_tcp())
      out = run_pipeline(obs, cad_cache, quiet) if obs else {}
      if "T_cad_to_cam" not in out:
        raise SystemExit("Could not find the object; pass --target x,y,z")
      T_bc = tcp_cam()
      c = out["T_cad_to_cam"][:3, :3] @ cad_cache[out["final_object"]].center + \
          out["T_cad_to_cam"][:3, 3]
      target = T_bc[:3, :3] @ c + T_bc[:3, 3]
      print(f"[+] Object center (base frame): {np.round(target, 4).tolist()}")
    else:
      target = np.asarray(parse_list(a.target))

    def go(t, az, label):
      Tc = look_at(target, t, az, a.dist)
      if T_cam_tool is None:   # manual dry-run without hand-eye: describe the view
        input(f"[manual] Move the camera to tilt {t:g} deg, azimuth {az:g} deg, "
              f"{a.dist:.2f} m from the object, then press Enter ")
        return
      Tt = Tc @ T_cam_tool
      check_target(Tc, Tt, target, a.min_dist, a.min_z, a.max_reach, tool_tip)
      robot.move_tcp(Tt, label)

    # ---------------------------------------------------------- reference
    ref_rec = None
    if a.label == "reference":
      rt, raz = parse_list(a.ref_view)
      cand = [(rt, raz)] + [(t, az) for t, az, _ in views if (t, az) != (rt, raz)]
      for t, az in cand:
        try:
          go(t, az, f"reference view tilt {t:g} az {az:g}")
        except ValueError as e:
          print(f"  [-] view skipped: {e}")
          continue
        T_bt = robot.get_tcp()
        obs = cam.capture(T_bt)
        if obs is None:
          print("  [-] nothing detected here, trying the next view")
          continue
        out = run_pipeline(obs, cad_cache, quiet)
        if "T_cad_to_cam" not in out or out.get("final_object") != a.object:
          print("  [-] no pose / wrong class here, trying the next view")
          continue
        print(f"  C={out['C']:.3f} {out['state']} {out.get('reason', '')}")
        ok = "y" if a.assume_yes else show_and_ask(obs, out, cad, gui)
        if ok == "y":
          ref_rec = dict(T_bt=T_bt, T_cc=out["T_cad_to_cam"])
          break
      if ref_rec is None:
        raise SystemExit("No reference pose accepted; try --label manual.")

    # ---------------------------------------------------------- sweep (capture)
    recs = []
    for i, (t, az, rep) in enumerate(views):
      print(f"\n[{i+1}/{len(views)}] tilt {t:g} az {az:g} rep {rep}")
      try:
        go(t, az, f"view {i+1}")
      except ValueError as e:
        print(f"  [-] view skipped: {e}")
        continue
      T_bt = robot.get_tcp()
      obs = cam.capture(T_bt)
      base = dict(object=a.object, color=a.color, view=i, tilt_deg=t,
                  azimuth_deg=az, rep=rep, tcp_pose=json.dumps(
                      [round(v, 6) for v in to_ur_pose(T_bt)]),
                  timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
      if obs is None:
        writer.writerow(dict(base, sample="real", state="no detection",
                             label="wrong", label_source="no detection"))
        print("  [-] nothing detected")
        continue
      out = run_pipeline(obs, cad_cache, quiet)
      print(f"  {out.get('final_object', obs.object_key)}: C={out.get('C', 0):.3f} "
            f"{out['state']} {out.get('reason', '')}")
      if a.label == "manual":
        k = show_and_ask(obs, out, cad, gui)
        out["_manual"] = dict(y="correct", n="wrong", b="borderline", s="skip")[k]
      recs.append(dict(base=base, obs=obs, out=out, T_bt=T_bt))

    # ---------------------------------------------------------- hand-eye
    X, P_mean = T_tool_cam, None
    if a.label == "reference" and not a.no_refine_hand_eye:
      good = [r for r in recs if "T_cad_to_cam" in r["out"]
              and r["out"].get("final_object") == a.object]
      sol = solve_hand_eye([ref_rec["T_bt"]] + [r["T_bt"] for r in good],
                           [ref_rec["T_cc"]] + [r["out"]["T_cad_to_cam"] for r in good],
                           cad, T_tool_cam, 0)
      if sol is None:
        print("\n[!] Hand-eye refinement not possible here (hemisphere, or fewer "
              "than 5 consistent views); labels use --hand-eye as given.")
      else:
        X, inl, res, P_mean = sol
        d_t = np.linalg.norm(X[:3, 3] - T_tool_cam[:3, 3]) * 1000
        d_r = _rot_deg(X[:3, :3], T_tool_cam[:3, :3])
        he_path = os.path.join(a.out, f"handeye_refined_{a.object}_{stamp}.npy")
        np.save(he_path, X)
        print(f"\n[+] Hand-eye solved from {int(inl.sum())}/{len(inl)} consistent "
              f"views (median residual {np.median(res[inl, 0]):.1f} mm, "
              f"{np.median(res[inl, 1]):.2f} deg).")
        print(f"    It differs from --hand-eye by {d_t:.1f} mm and {d_r:.1f} deg.")
        print(f"    t = {np.round(X[:3, 3] * 1000, 1).tolist()} mm. Saved to {he_path}")
        if isinstance(cam, SimCamera):
          print(f"[sim] refined vs true mount: "
                f"{np.linalg.norm(X[:3, 3] - cam.X[:3, 3]) * 1000:.2f} mm, "
                f"{_rot_deg(X[:3, :3], cam.X[:3, :3]):.2f} deg")
    # ground truth: the mean over all consistent views when the hand-eye was
    # solved (less noise than one view), else the view you approved
    T_base_cad_ref = P_mean if P_mean is not None else \
        (ref_rec["T_bt"] @ X @ ref_rec["T_cc"] if ref_rec else None)
    if isinstance(cam, SimCamera) and T_base_cad_ref is not None:
      e = pose_error(cam.T_base_obj, T_base_cad_ref, cad)
      print(f"[sim] reference pose error vs truth: {e[0]:.2f} deg, {e[1]:.2f} mm")

    # ---------------------------------------------------------- labels + injections
    for rec in recs:
      base, obs, out, T_bt = rec["base"], rec["obs"], rec["out"], rec["T_bt"]
      fin = out.get("final_object", obs.object_key)
      row = dict(base, sample="real", perturbation="", yolo_object=obs.object_key,
                 final_object=fin, C=round(out.get("C", 0.0), 4),
                 IB=round(out.get("IB", 0.0), 4), IM=round(out.get("IM", 0.0), 4),
                 margin=round(out.get("margin", float("nan")), 4),
                 class_margin=round(out.get("class_margin", float("nan")), 4),
                 axis_ambiguity_deg=out.get("axis_ambiguity_deg"),
                 state=out["state"], reason=out.get("reason", ""),
                 used_fallback=out.get("used_fallback"), chosen=out.get("chosen"),
                 UD=round(out.get("UD", float("nan")), 4),
                 caught_by_other_check=caught_by_other_check(out, params))
      wrong_class = fin != a.object
      T_true_cam = None
      if "T_cad_to_cam" not in out:
        row.update(label="wrong", label_source="no pose")
      else:
        row["T_cad_to_cam"] = json.dumps(np.round(out["T_cad_to_cam"], 6).tolist())
        if a.label == "reference":
          T_true_cam = np.linalg.inv(T_bt @ X) @ T_base_cad_ref
          e_r, e_t = pose_error(T_true_cam, out["T_cad_to_cam"], cad)
          row.update(rot_err_deg=round(e_r, 2), trans_err_mm=round(e_t, 2),
                     label=label_of(e_r, e_t, wrong_class), label_source="reference")
        else:
          lab = out["_manual"]
          if wrong_class and lab == "correct":
            lab = "wrong"
          row.update(label=lab, label_source="manual")
          if lab == "correct":
            T_true_cam = out["T_cad_to_cam"]
      if isinstance(cam, SimCamera) and "T_cad_to_cam" in out:
        tr, tt = pose_error(cam.true_pose_cam(T_bt), out["T_cad_to_cam"], cad)
        row.update(true_rot_err_deg=round(tr, 2), true_trans_err_mm=round(tt, 2))
      if not cam.simulated and "T_cad_to_cam" in out:
        d = os.path.join(a.out, "trials")
        with contextlib.redirect_stdout(io.StringIO()):
          vp.save_trial(obs, out, root=d)
        row["trial_dir"] = os.path.join(d, f"{obs.object_key}_{obs.meta['timestamp']}")
      writer.writerow(row)
      print(f"  view {base['view']} tilt {base['tilt_deg']:g} az {base['azimuth_deg']:g}: "
            f"C={row['C']:.3f} {row['state']} | label {row['label']} "
            f"{row.get('rot_err_deg', '')} deg {row.get('trans_err_mm', '')} mm")
      # injected errors around the true pose (only when this view saw it right)
      if not a.no_inject and T_true_cam is not None and not wrong_class \
          and row["label"] == "correct":
        with contextlib.redirect_stdout(io.StringIO()):
          inj = injected_rows(obs, out.get("scan"), cad, params, T_true_cam)
        for r in inj:
          writer.writerow(dict(base, yolo_object=obs.object_key,
                               final_object=a.object, **r))
      fcsv.flush()
  except KeyboardInterrupt:
    print("\n[!] Stopped; rows so far are saved.")
  finally:
    fcsv.close()
    robot.close()
    cam.close()
  print(f"\n[+] Wrote {csv_path}\n    Next: python code/vgap/recommend_tau_c.py "
        f"{os.path.join(a.out, 'calib_*.csv')}")


if __name__ == "__main__":
  main()
