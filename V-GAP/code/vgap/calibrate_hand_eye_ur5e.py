"""Eye-in-hand calibration of the wrist RealSense on the UR5e with one ArUco marker.

Tape one ArUco marker flat on the table (it must not move during the run).
The script visits 15-25 views of it, logs the robot TCP pose and the marker
pose in the camera at each view, and solves AX = XB with OpenCV's
cv2.calibrateHandEye (Tsai, Park, Daniilidis, plus Horaud).

Output: T_flange_cam.npy (4x4, meters, camera pose in the TCP frame), same
convention as now:  p_base = T_base_tcp @ T_flange_cam @ p_cam.
The camera frame is the RealSense COLOR optical frame, which is the frame the
pipeline's aligned depth lives in. Keep the pendant's active TCP the same as
in the gate runs (the flange): the script reads getActualTCPPose().
An existing output file is first copied to T_flange_cam_backup_<time>.npy.

Verification: the marker is static, so T_base_tcp @ X @ T_cam_marker must
give the same marker pose from every view. The report lists, per method and
for the old matrix, the spread of the marker position (mm, from the mean)
and orientation (deg). The method with the lowest median position spread is
saved (or force one with --method). Views more than max(--outlier-mm,
3 x median) off are dropped once and the solve is repeated.

Modes (--mode):
  auto    (default) Start with the camera roughly 30-40 cm above the marker,
          looking down. The script finds the marker with the old matrix
          (--hand-eye-init, a few cm off is fine), then orbits it: tilt
          0 deg with three camera rolls, then rings at --tilts (max 45 deg;
          60 collided before) with alternating roll. Every moveL is slow
          (5 cm/s) and needs Enter unless --no-confirm. The tool must stay
          --min-clear above the marker. Hand on the e-stop.
  manual  Freedrive (teach mode). Move the arm by hand, let go, press Enter
          to capture. Aim for tilts 0-45 deg from several sides AND some
          rotation about the camera axis. 'd' = done, 'q' = quit.
Add --plan-only (auto) to print the planned TCP poses without moving.

Marker: --dict auto tries the common dictionaries; --id picks one marker
(default: the only one seen). --marker-size is the black square's edge in
meters (measure the print, not the PDF value).

Examples (from the project root):
  python code/vgap/calibrate_hand_eye_ur5e.py --marker-size 0.08 --plan-only
  python code/vgap/calibrate_hand_eye_ur5e.py --marker-size 0.08 --dict 4x4_50 --id 0
  python code/vgap/calibrate_hand_eye_ur5e.py --marker-size 0.08 --mode manual
  # re-solve saved samples offline (no robot, no camera)
  python code/vgap/calibrate_hand_eye_ur5e.py --solve-only data/handeye/samples_<time>.json
Saves: data/handeye/samples_<time>.json (raw poses, for re-solving),
       data/handeye/report_<time>.txt, and the output matrix.
"""
import argparse
import contextlib
import json
import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np

if not hasattr(cv2, "calibrateHandEye") or not hasattr(cv2, "aruco"):
  raise SystemExit("needs OpenCV 4.7-4.x with aruco: "
                   "pip install 'opencv-contrib-python<5'")

UR_IP = "192.168.1.5"
WRIST_SERIAL = "337122075128"
DICTS = {"4x4_50": "DICT_4X4_50", "4x4_100": "DICT_4X4_100", "4x4_250": "DICT_4X4_250",
         "5x5_100": "DICT_5X5_100", "5x5_250": "DICT_5X5_250",
         "6x6_250": "DICT_6X6_250", "7x7_250": "DICT_7X7_250",
         "original": "DICT_ARUCO_ORIGINAL", "apriltag_36h11": "DICT_APRILTAG_36h11"}
METHODS = {"tsai": cv2.CALIB_HAND_EYE_TSAI, "park": cv2.CALIB_HAND_EYE_PARK,
           "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
           "horaud": cv2.CALIB_HAND_EYE_HORAUD, "andreff": cv2.CALIB_HAND_EYE_ANDREFF}


# ==============================================================================
# Geometry helpers (same as calibrate_tau_c_ur5e.py)
# ==============================================================================
def to_ur_pose(T):
  rvec, _ = cv2.Rodrigues(T[:3, :3])
  return [float(v) for v in T[:3, 3]] + [float(v) for v in rvec.ravel()]


def from_ur_pose(p):
  T = np.eye(4)
  T[:3, :3], _ = cv2.Rodrigues(np.asarray(p[3:], float))
  T[:3, 3] = p[:3]
  return T


def _rot_deg(Ra, Rb):
  return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def _rot_axis_angle(axis, deg):
  axis = np.asarray(axis, float)
  n = np.linalg.norm(axis)
  if n < 1e-12:
    return np.eye(3)
  return cv2.Rodrigues(axis / n * np.radians(deg))[0]


def plan_views(T_base_cam0, p_marker, dist, tilts, n_az, roll_deg):
  """Camera poses (base frame) orbiting the marker at distance dist.

  Each view tilts the start camera by the smallest rotation that points its
  optical axis at the marker from (tilt, azimuth), then rolls it about that
  axis. Staying close to the start orientation keeps the wrist in the same
  configuration (no flips between views).
  """
  R0 = T_base_cam0[:3, :3]
  z0 = R0[:, 2]
  views = []
  for roll in (0.0, roll_deg, -roll_deg):           # straight down, 3 rolls
    views.append((0.0, 0.0, roll))
  for i, tilt in enumerate(tilts):
    if tilt <= 0:
      continue
    n = n_az if tilt >= max(tilts) else max(4, n_az // 2)
    for k in range(n):
      az = 360.0 * k / n + (22.5 if i % 2 else 0.0)
      views.append((tilt, az, roll_deg if k % 2 else -roll_deg))
  out = []
  for tilt, az, roll in views:
    t, a = np.radians(tilt), np.radians(az)
    cpos = p_marker + dist * np.array([np.sin(t) * np.cos(a),
                                       np.sin(t) * np.sin(a), np.cos(t)])
    z = (p_marker - cpos) / np.linalg.norm(p_marker - cpos)
    ang = np.degrees(np.arccos(np.clip(z0 @ z, -1, 1)))
    R = _rot_axis_angle(np.cross(z0, z), ang) @ R0
    R = R @ _rot_axis_angle([0, 0, 1], roll)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cpos
    out.append((f"tilt{tilt:.0f}_az{az:.0f}_roll{roll:+.0f}", T))
  return out


# ==============================================================================
# Marker detection
# ==============================================================================
class MarkerDetector:
  def __init__(self, dict_name, marker_id, size, K, dist_coeffs):
    self.size, self.K, self.D = size, K, dist_coeffs
    self.id = marker_id
    self.dict_name = dict_name
    self.params = cv2.aruco.DetectorParameters()
    self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    self.det = None if dict_name == "auto" else self._make(dict_name)
    s = size / 2  # marker corners in the marker frame (IPPE_SQUARE order)
    self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

  def _make(self, name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTS[name]))
    return cv2.aruco.ArucoDetector(d, self.params)

  def corners(self, color):
    """4x2 corners of the chosen marker, or None."""
    if self.det is None:                       # --dict auto: lock on first hit
      for name in DICTS:
        c, ids, _ = self._make(name).detectMarkers(color)
        if ids is not None:
          self.det, self.dict_name = self._make(name), name
          print(f"[+] marker dictionary: {name}, ids seen {ids.ravel().tolist()}")
          break
      else:
        return None
    c, ids, _ = self.det.detectMarkers(color)
    if ids is None:
      return None
    ids = ids.ravel().tolist()
    if self.id is None:
      if len(ids) > 1:
        raise SystemExit(f"several markers seen {ids}: pick one with --id")
      self.id = ids[0]
      print(f"[+] using marker id {self.id}")
    if self.id not in ids:
      return None
    return c[ids.index(self.id)].reshape(4, 2)

  def pose(self, corners):
    """T_cam_marker from averaged corners + reprojection error (px).

    IPPE_SQUARE gives two solutions; near-frontal views can flip between
    them, so the ratio of their errors is returned too (< 2 = ambiguous).
    """
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        self.obj, corners.astype(np.float32), self.K, self.D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvecs[order[0]])[0]
    T[:3, 3] = tvecs[order[0]].ravel()
    ratio = float(errs[order[1]] / max(errs[order[0]], 1e-9)) if n > 1 else np.inf
    return T, float(errs[order[0]]), ratio


class ColorCamera:
  """Wrist RealSense color stream only (the marker needs no depth)."""

  def __init__(self, serial, width=640, height=480, fps=30):
    import pyrealsense2 as rs
    self.pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    prof = self.pipeline.start(cfg)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    self.K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1]], float)
    self.D = np.array(it.coeffs, float)
    for _ in range(30):                        # let auto exposure settle
      self.pipeline.wait_for_frames()
    print(f"[+] RealSense {serial} color {width}x{height}, fx {it.fx:.1f}")

  def frame(self):
    while True:
      c = self.pipeline.wait_for_frames().get_color_frame()
      if c:
        return np.asanyarray(c.get_data()).copy()

  def close(self):
    self.pipeline.stop()


def measure(cam, det, n_frames=15, settle_s=1.0, min_hits=0.7):
  """Average marker corners over n_frames still frames -> (T_cam_marker, info)."""
  t0 = time.time()
  while time.time() - t0 < settle_s:          # flush frames taken while moving
    cam.frame()
  hits = []
  for _ in range(n_frames):
    c = det.corners(cam.frame())
    if c is not None:
      hits.append(c)
  if len(hits) < min_hits * n_frames:
    return None, f"marker seen in {len(hits)}/{n_frames} frames"
  hits = np.array(hits)
  jitter = float(np.max(np.std(hits, axis=0)))
  T, err, ratio = det.pose(hits.mean(axis=0))
  return T, dict(reproj_px=err, ippe_ratio=ratio, jitter_px=jitter,
                 dist_m=float(np.linalg.norm(T[:3, 3])), hits=len(hits))


# ==============================================================================
# Robot
# ==============================================================================
class UR5e:
  def __init__(self, ip, speed=0.05, acc=0.10, confirm=True, move=True):
    import rtde_receive
    self.r = rtde_receive.RTDEReceiveInterface(ip)
    self.c = None
    if move:
      import rtde_control
      self.c = rtde_control.RTDEControlInterface(ip)
    self.speed, self.acc, self.confirm = speed, acc, confirm

  def get_tcp(self):
    return from_ur_pose(self.r.getActualTCPPose())

  def move_tcp(self, T_base_tool, label=""):
    p = to_ur_pose(T_base_tool)
    print(f"[robot] {label} moveL to {np.round(p, 4).tolist()}")
    if not self.c.isPoseWithinSafetyLimits(p):
      print("        outside the safety limits -> skipped")
      return False
    if self.confirm:
      a = input("        Enter = move, s = skip, q = stop ").strip().lower()
      if a == "q":
        raise KeyboardInterrupt("stopped by operator")
      if a == "s":
        return False
    if not self.c.moveL(p, self.speed, self.acc):
      raise RuntimeError("moveL failed (protective stop / unreachable?)")
    return True

  def teach(self, on):
    if on:
      self.c.teachMode()
    else:
      self.c.endTeachMode()

  def close(self):
    with contextlib.suppress(Exception):
      self.c.stopScript()


# ==============================================================================
# Solve + verify
# ==============================================================================
def marker_in_base(T_bts, T_cms, X):
  return [Tb @ X @ Tc for Tb, Tc in zip(T_bts, T_cms)]


def spread(T_bts, T_cms, X):
  """Per-view marker position error (mm) and orientation error (deg) vs mean."""
  Ms = marker_in_base(T_bts, T_cms, X)
  P = np.array([M[:3, 3] for M in Ms])
  pos = np.linalg.norm(P - P.mean(axis=0), axis=1) * 1000
  U, _, Vt = np.linalg.svd(sum(M[:3, :3] for M in Ms))  # chordal mean rotation
  Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
  rot = np.array([_rot_deg(Rm, M[:3, :3]) for M in Ms])
  return pos, rot, P.mean(axis=0)


def solve(T_bts, T_cms, method):
  R, t = cv2.calibrateHandEye(
      [T[:3, :3] for T in T_bts], [T[:3, 3] for T in T_bts],
      [T[:3, :3] for T in T_cms], [T[:3, 3] for T in T_cms], method=METHODS[method])
  X = np.eye(4)
  X[:3, :3], X[:3, 3] = R, t.ravel()
  return X


def solve_all(T_bts, T_cms, X_old, methods, outlier_mm, outlier_deg, log):
  """Solve with every method; drop outlier views once; return {method: X}."""
  keep = np.arange(len(T_bts))
  ref = solve(T_bts, T_cms, "park")
  pos, rot, _ = spread(T_bts, T_cms, ref)
  bad = (pos > max(outlier_mm, 3 * np.median(pos))) | \
        (rot > max(outlier_deg, 3 * np.median(rot)))
  if bad.any() and len(T_bts) - bad.sum() >= 8:
    log(f"dropped {bad.sum()} outlier view(s): {np.flatnonzero(bad).tolist()} "
        f"(pos {np.round(pos[bad], 1).tolist()} mm, rot {np.round(rot[bad], 1).tolist()} deg)")
    keep = np.flatnonzero(~bad)
  Tb = [T_bts[i] for i in keep]
  Tc = [T_cms[i] for i in keep]
  res = {}
  log(f"\n{len(keep)} views used.  Marker spread from the mean (lower = better):")
  log(f"{'method':<12}{'pos med':>9}{'pos max':>9}{'rot med':>9}{'rot max':>9}"
      f"   t (mm)                 R vs old (deg)")
  rows = []
  for m in methods:
    try:
      rows.append((m, solve(Tb, Tc, m)))
    except cv2.error as e:
      log(f"{m:<12}failed ({str(e).strip().splitlines()[-1][:60]})")
  if X_old is not None:
    rows.append(("OLD matrix", X_old))
  for m, X in rows:
    pos, rot, _ = spread(Tb, Tc, X)
    dR = _rot_deg(X[:3, :3], X_old[:3, :3]) if X_old is not None else float("nan")
    log(f"{m:<12}{np.median(pos):>8.1f} {pos.max():>8.1f} {np.median(rot):>8.2f} "
        f"{rot.max():>8.2f}   {np.round(X[:3, 3] * 1000, 1).tolist()!s:<22} {dR:.2f}")
    if m != "OLD matrix":
      res[m] = (X, float(np.median(pos)))
  return res, keep


def save_matrix(X, out, log):
  if os.path.exists(out):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(os.path.dirname(out) or ".",
                       f"T_flange_cam_backup_{stamp}.npy")
    os.makedirs(os.path.dirname(bak), exist_ok=True)
    shutil.copy2(out, bak)
    log(f"[+] old matrix backed up to {bak}")
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  np.save(out, X)
  log(f"[+] saved {out}")


# ==============================================================================
def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--marker-size", type=float, help="black square edge (m), measured")
  ap.add_argument("--dict", default="auto", choices=["auto"] + list(DICTS))
  ap.add_argument("--id", type=int, default=None, help="marker id (default: the only one)")
  ap.add_argument("--mode", default="auto", choices=["auto", "manual"])
  ap.add_argument("--robot-ip", default=UR_IP)
  ap.add_argument("--serial", default=WRIST_SERIAL, help="wrist RealSense serial")
  ap.add_argument("--hand-eye-init", default="T_flange_cam.npy",
                  help="current matrix: used to aim the auto views and as the 'old' baseline")
  ap.add_argument("--out", default="T_flange_cam.npy")
  ap.add_argument("--dist", type=float, default=0.35, help="camera-marker distance (m)")
  ap.add_argument("--tilts", default="20,35,45", help="ring tilts (deg), max 45")
  ap.add_argument("--n-az", type=int, default=8, help="views on the widest ring")
  ap.add_argument("--roll", type=float, default=25.0, help="camera roll per view (deg)")
  ap.add_argument("--min-clear", type=float, default=0.15,
                  help="TCP must stay this far above the marker (m)")
  ap.add_argument("--speed", type=float, default=0.05)
  ap.add_argument("--no-confirm", action="store_true")
  ap.add_argument("--plan-only", action="store_true")
  ap.add_argument("--frames", type=int, default=15, help="frames averaged per view")
  ap.add_argument("--method", default="best", choices=["best"] + list(METHODS))
  ap.add_argument("--outlier-mm", type=float, default=5.0)
  ap.add_argument("--outlier-deg", type=float, default=2.0)
  ap.add_argument("--solve-only", default=None, help="samples json from an earlier run")
  ap.add_argument("--no-save", action="store_true", help="report only, keep the old matrix")
  a = ap.parse_args()

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  os.makedirs("data/handeye", exist_ok=True)
  lines = []

  def log(s=""):
    print(s)
    lines.append(s)

  X_old = np.load(a.hand_eye_init) if os.path.exists(a.hand_eye_init) else None
  if X_old is None:
    log(f"[!] {a.hand_eye_init} not found: no old baseline")

  if a.solve_only:
    S = json.load(open(a.solve_only))
    T_bts = [np.array(s["T_base_tcp"]) for s in S["samples"]]
    T_cms = [np.array(s["T_cam_marker"]) for s in S["samples"]]
  else:
    if not a.marker_size:
      raise SystemExit("--marker-size (m) is required")
    if max(float(t) for t in a.tilts.split(",")) > 45:
      raise SystemExit("tilts above 45 deg are not allowed (60 collided before)")
    robot = UR5e(a.robot_ip, a.speed, confirm=not a.no_confirm,
                 move=not a.plan_only)
    cam = ColorCamera(a.serial)
    det = MarkerDetector(a.dict, a.id, a.marker_size, cam.K, cam.D)
    T_bts, T_cms, samples = [], [], []

    def grab(label):
      time.sleep(0.5)
      T_bt = robot.get_tcp()
      T_cm, info = measure(cam, det, a.frames)
      if T_cm is None:
        print(f"    {label}: skipped ({info})")
        return
      T_bts.append(T_bt)
      T_cms.append(T_cm)
      samples.append(dict(label=label, T_base_tcp=T_bt.tolist(),
                          T_cam_marker=T_cm.tolist(), **info))
      flag = "  AMBIGUOUS (near frontal)" if info["ippe_ratio"] < 2 else ""
      print(f"    {label}: #{len(T_bts)}  reproj {info['reproj_px']:.2f}px  "
            f"jitter {info['jitter_px']:.2f}px  dist {info['dist_m']:.3f}m{flag}")

    try:
      if a.mode == "manual":
        print("Freedrive ON. Move the arm, let go, then Enter = capture, "
              "d = done, q = quit.  Aim for 15-25 views.")
        robot.teach(True)
        while True:
          k = input(f"[{len(T_bts)} views] ").strip().lower()
          if k == "q":
            raise KeyboardInterrupt
          if k == "d":
            break
          robot.teach(False)
          grab(f"manual{len(samples)}")
          robot.teach(True)
        robot.teach(False)
      else:
        if X_old is None:
          raise SystemExit("auto mode needs --hand-eye-init to aim the views")
        T0 = robot.get_tcp()
        T_cm0, info = measure(cam, det, a.frames)
        if T_cm0 is None:
          raise SystemExit(f"marker not visible from the start pose ({info})")
        p_m = (T0 @ X_old @ T_cm0)[:3, 3]
        log(f"marker (old matrix) at {np.round(p_m, 4).tolist()} m in base")
        tilts = [float(t) for t in a.tilts.split(",")]
        plan = plan_views(T0 @ X_old, p_m, a.dist, tilts, a.n_az, a.roll)
        print(f"{len(plan)} views planned")
        for label, T_bc in plan:
          T_bt = T_bc @ np.linalg.inv(X_old)
          if T_bt[2, 3] < p_m[2] + a.min_clear:
            print(f"    {label}: TCP {T_bt[2, 3]:.3f} m too low -> skipped")
            continue
          if a.plan_only:
            print(f"    {label}: TCP {np.round(to_ur_pose(T_bt), 4).tolist()}")
            continue
          if robot.move_tcp(T_bt, label):
            grab(label)
        if not a.plan_only:
          robot.move_tcp(T0, "back to start")
    except KeyboardInterrupt:
      print("\nstopped; solving with the views so far")
    finally:
      cam.close()
      if robot.c is not None:
        robot.close()
    if a.plan_only:
      return
    path = f"data/handeye/samples_{stamp}.json"
    json.dump(dict(marker_size=a.marker_size, dict=det.dict_name, id=det.id,
                   K=cam.K.tolist(), D=cam.D.tolist(), samples=samples),
              open(path, "w"), indent=1)
    log(f"[+] {len(samples)} views saved to {path}")

  if len(T_bts) < 8:
    raise SystemExit(f"only {len(T_bts)} views; need >= 8 (15-25 recommended)")
  rots = [_rot_deg(T_bts[0][:3, :3], T[:3, :3]) for T in T_bts]
  log(f"TCP rotation range vs first view: max {max(rots):.0f} deg "
      f"(< 20 deg gives a poor rotation estimate)")

  res, keep = solve_all(T_bts, T_cms, X_old, ["tsai", "park", "daniilidis", "horaud"],
                        a.outlier_mm, a.outlier_deg, log)
  pick = min(res, key=lambda m: res[m][1]) if a.method == "best" else a.method
  X = res[pick][0]
  _, _, p_mean = spread([T_bts[i] for i in keep], [T_cms[i] for i in keep], X)
  log(f"\nchosen: {pick}")
  log(f"T_flange_cam =\n{np.array2string(X, precision=5, suppress_small=True)}")
  log(f"camera offset in flange frame: {np.round(X[:3, 3] * 1000, 1).tolist()} mm")
  if X_old is not None:
    log(f"change vs old: {np.linalg.norm(X[:3, 3] - X_old[:3, 3]) * 1000:.1f} mm, "
        f"{_rot_deg(X[:3, :3], X_old[:3, :3]):.2f} deg")
  log(f"marker center in base: {np.round(p_mean, 4).tolist()} m "
      f"(check: jog the TCP tip onto the marker center and compare)")
  if not a.no_save:
    save_matrix(X, a.out, log)
  open(f"data/handeye/report_{stamp}.txt", "w").write("\n".join(lines) + "\n")
  print(f"[+] report: data/handeye/report_{stamp}.txt")


if __name__ == "__main__":
  main()
