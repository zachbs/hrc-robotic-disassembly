"""V-GAP grasp trials on the UR5e + Robotiq 2F gripper.

Does the gate turn into more successful grasps? Each trial captures from a
fixed view, estimates the pose, plans a grasp from the CAD, grasps, lifts,
holds 5 s, and you say whether the part is still held.

Two methods, run on the SAME placements and views (paired design):
  vgap   full V-GAP. Up to --vgap-bursts bursts at the view (= the same-view
         retry); the first Valid pose is grasped. No Valid pose -> the robot
         does NOT grasp (outcome no_grasp_gate).
  plain  plain RANSAC+ICP for YOLO's class (the pose before the hypothesis
         check, table fallback and class check, as in the gate eval's
         section K). Always grasped. What the gate would have said about that
         pose is logged too (plain_gate_would_pass), so you can show that
         the gate predicts which plain grasps fail.

Success rule (fixed before the trials): the part is lifted off the table and
is still held in the gripper at the end of a 5 s hold, judged by the operator
(y/n), with the Robotiq object-detect status logged as a check.

Per object: 5 setups (placement, yaw, view tilt) x 2 reps x 2 methods = 20
trials, 10 per method. Order: objects shuffled, setups shuffled, and in each
rep the method order is random. The plan is saved first (plan.csv), so
--resume continues after a crash or protective stop.

Grasps (CAD frame, mm; picked from the CAD geometry, see GRASPS below):
  L      across one arm (33.3 mm), 15 mm past the inner corner on the arm, so the
         fingers clear the other arm by ~18 mm (moved out from 3 mm after the
         11/03 pilot came within ~3 mm of the other arm). The center of mass
         (-13.3, -13.3) is in the notch, so it cannot be gripped directly; this is the
         closest spot with clearance. Both arms are listed (symmetry), and the
         script picks the one that needs the least wrist turn.
  T      across the stem (33.3 mm) at y = -12, ~15 mm below the crossbar
         (pad 22 mm wide), 18 mm from the center of mass (0, 13.3). Backup:
         across the crossbar ends.
  gearbox  2F-140: across the 90 mm body width at the center (center of mass),
         fingertips 25 mm below the top. 2F-85 (85 mm stroke < 90 mm body):
         only a pinch on an end lip, 3 mm from the body (risky, off unless
         --allow-experimental).
  hemisphere  2F-140: across the dome 8 mm above the base (98.7 mm chord,
         contact normal 9 deg from horizontal, inside the friction cone),
         any yaw. 2F-85: none (100 mm > 85 mm; a grip high on the dome
         slopes 44 deg and squeezes out).
All grasps come straight down along the part's top-face normal (top face
up; a propped part tilts the approach by the prop angle, limit
--max-approach-tilt). Fingertips stay >= --tip-clear above the table, or the
grasp is made shallower (down to --min-contact); otherwise it is refused
(no_grasp_unsafe, counted as a failure).

Safety:
  * slow moveL (--speed), slower final approach (--approach-speed);
  * Enter before every move (--confirm all; 'grasp' asks only before the
    descent; type q to stop);
  * guarded descent: the move stops if the wrist force sensor reads more
    than --contact-force N (outcome fail_collision), then backs off;
  * the grasp must be within --max-grasp-offset of the detected object;
  * protective stop: the trial is logged as protective_stop, you clear it
    on the pendant, the script reconnects and goes on (or --resume later).
  * The operator may abort ONLY for safety (would hit table / fixture);
    an abort counts as a failure for that method (fixed before the trials).

Modes:
  --mode check-grasps   offline: fingers vs CAD clearance for each grasp
  --mode setup          on the robot: measure the fingertip offset and the
                        finger closing axis (saves data/grasp_trials/setup.json)
  --mode run            the trials (--robot sim tests everything in simulation)
  --mode report         summary over trials.csv files

UNTESTED ON HARDWARE. Watch the first trials with a hand on the stop.
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os
import random
import socket
import sys
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vgap_pipeline as vp  # noqa: E402
import calibrate_tau_c_ur5e as cal  # noqa: E402
import vgap_trials_ur5e as tr  # noqa: E402
# importing the gate eval installs its spy that keeps the plain RANSAC+ICP pose
import vgap_gate_eval_ur5e as ge  # noqa: E402

# ==============================================================================
# Gripper models (mm). Finger sizes are approximate (Robotiq drawings),
# used only for the clearance checks; --pad-width etc. override them.
# ==============================================================================
GRIPPERS = {
    "2f85": dict(stroke=85.0, pad_w=22.0, finger_t=10.0, pad_len=30.0, finger_len=55.0),
    "2f140": dict(stroke=140.0, pad_w=27.0, finger_t=10.0, pad_len=35.0, finger_len=70.0),
}

# ==============================================================================
# Grasp library, CAD frame of each STL in vgap_pipeline (mm).
#   c      grasp center (x, y) in the CAD frame
#   u      finger closing direction in the CAD x-y plane
#   top    z of the top of the gripped feature (approach comes down -z)
#   depth  how far the fingertips go below `top`
#   width  feature width between the fingers
#   open   planned opening (clamped to the stroke - 3 mm)
#   prio   lower = preferred
#   yaw_free  any closing direction about z works (hemisphere)
#   mirror  part is symmetric about z = 0, so the same grasp also exists
#           from the -z side (used when V-GAP's pose has -z up)
# ==============================================================================
GRASPS = {
    "l_block": [   # 100 x 100 x 26.7, arms 33.3; arm A along x at y in [-50, -16.7]
        dict(name="L arm A", c=(15.0, -33.33), u=(0, 1), top=13.33, depth=16.0,
             width=33.33, open=70.0, prio=0, grippers=("2f85", "2f140")),
        dict(name="L arm B", c=(-33.33, 15.0), u=(1, 0), top=13.33, depth=16.0,
             width=33.33, open=70.0, prio=0, grippers=("2f85", "2f140")),
    ],
    "t_block": [   # crossbar y in [16.7, 50], stem x in [-16.7, 16.7]
        dict(name="T stem", c=(0.0, -12.0), u=(1, 0), top=13.33, depth=16.0,
             width=33.33, open=70.0, prio=0, grippers=("2f85", "2f140")),
        dict(name="T bar +x", c=(40.0, 33.33), u=(0, 1), top=13.33, depth=16.0,
             width=33.33, open=66.0, prio=1, grippers=("2f85", "2f140")),
        dict(name="T bar -x", c=(-40.0, 33.33), u=(0, 1), top=13.33, depth=16.0,
             width=33.33, open=66.0, prio=1, grippers=("2f85", "2f140")),
    ],
    "gearbox": [   # body x in [-55.8, 55.8], y +-45, z +-32.4; lips z +-9 to x +-76.5
        dict(name="gearbox body", c=(0.0, 0.0), u=(0, 1), top=32.4, depth=25.0,
             width=90.0, open=130.0, prio=0, grippers=("2f140",)),
        dict(name="gearbox lip +x", c=(69.0, 0.0), u=(0, 1), top=9.0, depth=12.0,
             width=21.0, open=50.0, prio=1, grippers=("2f85",), experimental=True),
        dict(name="gearbox lip -x", c=(-69.0, 0.0), u=(0, 1), top=9.0, depth=12.0,
             width=21.0, open=50.0, prio=1, grippers=("2f85",), experimental=True),
    ],
    "hemisphere": [   # base z = -25, dome top z = +25, R 50
        dict(name="hemisphere low", c=(0.0, 0.0), u=(1, 0), top=25.0, depth=42.0,
             width=98.7, open=134.0, prio=0, grippers=("2f140",), yaw_free=True,
             mirror=False),
    ],
}
MIRROR_DEFAULT = dict(l_block=True, t_block=True, gearbox=True, hemisphere=False)

SETUPS_DEFAULT = "flat:0:30,flat:90:45,flat:180:0,propped:0:30,propped:90:45"
OBJECT_ORDER = ["l_block", "t_block", "gearbox", "hemisphere"]

OUTCOMES = ["success", "fail_dropped", "fail_miss", "fail_collision", "no_grasp_gate",
            "no_grasp_unsafe", "no_detection", "aborted_safety", "protective_stop"]
ATTEMPTED = {"success", "fail_dropped", "fail_miss", "fail_collision",
             "aborted_safety", "protective_stop"}

FIELDS = [
    "trial_id", "object", "color", "setup", "placement", "yaw_label", "tilt_deg", "rep",
    "method", "order", "timestamp", "outcome", "held_5s", "shifted", "note",
    "bursts", "yolo_object", "final_object", "class_swapped", "gate_state", "gate_reason",
    "C", "F3", "margin", "plain_C", "plain_F3", "plain_gate_would_pass",
    "plain_vs_vgap_rot_deg", "plain_vs_vgap_trans_mm",
    "pose_used", "obj_x_mm", "obj_y_mm", "obj_z_mm", "obj_rx", "obj_ry", "obj_rz",
    "grasp_name", "grasp_mirrored", "approach_tilt_deg", "contact_depth_mm",
    "open_mm", "expected_width_mm", "tip_above_table_mm", "grasp_offset_mm",
    "grasp_tcp", "view_tcp", "obj_status_close", "width_close_mm",
    "obj_status_hold", "width_hold_mm", "max_force_N", "refused_reason",
    "runtime_s", "trial_s", "trial_dir", "sim_would_hold",
]


# ==============================================================================
# Grasp geometry
# ==============================================================================
def _unit(v):
  v = np.asarray(v, float)
  return v / np.linalg.norm(v)


def grasp_variants(obj, gripper, allow_exp=False, yaw_step=15.0):
  """Every grasp as (info, tip point m, approach unit, closing unit), CAD frame."""
  out = []
  for g in GRASPS[obj]:
    if gripper not in g["grippers"] or (g.get("experimental") and not allow_exp):
      continue
    mirror = g.get("mirror", MIRROR_DEFAULT[obj])
    us = [np.array([*g["u"], 0.0], float)]
    if g.get("yaw_free"):
      us = [np.array([np.cos(np.radians(t)), np.sin(np.radians(t)), 0.0])
            for t in np.arange(0, 180, yaw_step)]
    for side in ([1, -1] if mirror else [1]):
      tip = np.array([g["c"][0], g["c"][1], side * (g["top"] - g["depth"])]) / 1000.0
      a = np.array([0, 0, -side], float)       # comes down onto the top face
      for u in us:
        for sgn in (1, -1):                    # fingers swapped = same grasp
          out.append(dict(info=g, mirrored=side < 0, tip=tip, a=a, u=sgn * u,
                          top=side * g["top"] / 1000.0))
  return out


def finger_boxes(tip, a, u, open_m, gp, margin=0.0):
  """Two finger boxes (center, axes 3x3, half sizes) for a grasp; mm in gp."""
  v = np.cross(a, u)
  R = np.stack([u, v, a], 1)
  boxes = []
  t, w, L = gp["finger_t"] / 1000, gp["pad_w"] / 1000, gp["finger_len"] / 1000
  for s in (1, -1):
    c = tip + R @ np.array([s * (open_m / 2 + t / 2), 0, -L / 2])
    boxes.append((c, R, np.array([t / 2 + margin, w / 2 + margin, L / 2 + margin])))
  return boxes


def points_in_box(P, box):
  c, R, h = box
  q = (P - c) @ R
  return np.all(np.abs(q) <= h, axis=1)


def closing_contact(P, tip, a, u, open_m, gp):
  """Object points between the pads: (n points, width along u m, depth span m)."""
  v = np.cross(a, u)
  q = (P - tip) @ np.stack([u, v, a], 1)
  pad_len = gp["pad_len"] / 1000
  sel = (np.abs(q[:, 0]) < open_m / 2) & (np.abs(q[:, 1]) < gp["pad_w"] / 2000) & \
        (q[:, 2] <= 0.0005) & (q[:, 2] >= -pad_len)
  if sel.sum() < 10:
    return int(sel.sum()), 0.0, 0.0, 0.0
  qs = q[sel]
  return (int(sel.sum()), float(qs[:, 0].max() - qs[:, 0].min()),
          float(qs[:, 2].max() - qs[:, 2].min()), float(abs(qs[:, 0].mean())))


def cad_points(obj, n=40000):
  import open3d as o3d
  cfg = vp.OBJECT_CONFIGS[obj]
  mesh = o3d.io.read_triangle_mesh(vp.resolve_path(cfg["cad_path"]))
  mesh.scale(0.001 if cfg["cad_units"] == "mm" else 1.0, center=np.zeros(3))
  with contextlib.suppress(Exception):
    o3d.utility.random.seed(0)
  return np.asarray(mesh.sample_points_uniformly(n).points)


def check_grasps(gripper, gp, allow_exp):
  """Offline: clearance between the open fingers and the part, contact patch."""
  print(f"Gripper {gripper}: stroke {gp['stroke']} mm, pad {gp['pad_w']} mm wide, "
        f"finger {gp['finger_t']} mm thick, pad length {gp['pad_len']} mm\n")
  for obj in OBJECT_ORDER:
    P = cad_points(obj)
    vs = [v for v in grasp_variants(obj, gripper, allow_exp, yaw_step=90)
          if not v["mirrored"]]
    if not vs:
      print(f"{obj:11s} NO grasp for {gripper}"
            + (" (try --allow-experimental)" if any(g.get('experimental') and
               gripper in g['grippers'] for g in GRASPS[obj]) else ""))
      continue
    seen = set()
    for v in vs:
      g = v["info"]
      if (g["name"], tuple(np.round(np.abs(v["u"]), 3))) in seen:
        continue
      seen.add((g["name"], tuple(np.round(np.abs(v["u"]), 3))))
      open_m = min(g["open"], gp["stroke"] - 3) / 1000
      boxes = finger_boxes(v["tip"], v["a"], v["u"], open_m, gp)
      # clearance: grow the boxes until they touch the part
      clear = None
      for m in np.arange(0, 0.0301, 0.0005):
        b2 = finger_boxes(v["tip"], v["a"], v["u"], open_m, gp, margin=m)
        if any(points_in_box(P, b).any() for b in b2):
          clear = m
          break
      n, wid, dep, off = closing_contact(P, v["tip"], v["a"], v["u"], open_m, gp)
      inside = any(points_in_box(P, b).any() for b in boxes)
      bottom = P[:, 2].min()
      tip_up = (v["tip"][2] - bottom) * 1000
      print(f"{obj:11s} {g['name']:16s} open {open_m*1000:5.1f}  grip width "
            f"{wid*1000:5.1f} mm  contact depth {dep*1000:4.1f} mm  "
            f"finger clearance {'COLLIDES' if inside else ('>30' if clear is None else f'{clear*1000:.1f}')} mm  "
            f"tip {tip_up:4.1f} mm above part bottom  slack/side "
            f"{(open_m - wid) / 2 * 1000:4.1f} mm")


def plan_grasp(obj, T_bo, T_ref_tool, gp, a, P_obj, center_b):
  """Best safe grasp for a pose (base frame). Returns dict or (None, reason)."""
  R_bo, t_bo = T_bo[:3, :3], T_bo[:3, 3]
  P_b = P_obj @ R_bo.T + t_bo
  down = np.array([0, 0, -1.0])
  tip_tcp = np.array([a.tip_xy[0], a.tip_xy[1], a.tip_z])
  reasons, cands = [], []
  for v in grasp_variants(obj, a.gripper, a.allow_experimental):
    g = v["info"]
    a_b = R_bo @ v["a"]
    tilt = np.degrees(np.arccos(np.clip(a_b @ down, -1, 1)))
    if tilt > a.max_approach_tilt:
      reasons.append(f"approach tilt {tilt:.0f} deg")
      continue
    u_b = R_bo @ v["u"]
    u_b = _unit(u_b - (u_b @ a_b) * a_b)
    open_m = min(g["open"], gp["stroke"] - 3) / 1000
    # shallower if the fingertips would come too close to the table
    depth0 = g["depth"] / 1000
    ok = None
    for depth in np.arange(depth0, a.min_contact / 1000 - 1e-9, -0.001):
      tip_o = v["tip"] - v["a"] * (depth0 - depth)                 # back up = shallower
      tip_b = R_bo @ tip_o + t_bo
      boxes = finger_boxes(tip_b, a_b, u_b, open_m, gp)
      low = min((c + R @ (np.array(s) * h))[2] for c, R, h in boxes
                for s in [(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
      if low < a.table_z + a.tip_clear / 1000:
        continue
      if any(points_in_box(P_b, b).any() for b in
             finger_boxes(tip_b, a_b, u_b, open_m, gp, margin=a.finger_margin / 1000)):
        reasons.append(f"{g['name']}: fingers hit the part")
        break
      ok = (depth, tip_b, low)
      break
    if ok is None:
      reasons.append(f"{g['name']}: fingertips too close to the table")
      continue
    depth, tip_b, low = ok
    off = np.linalg.norm((tip_b - center_b)[:2]) if center_b is not None else 0.0
    if off > a.max_grasp_offset:
      reasons.append(f"{g['name']}: {off*1000:.0f} mm from the detected object")
      continue
    R_bt = np.eye(3)
    R_bt[:, 2] = a_b
    if a.close_axis == "y":
      R_bt[:, 1] = u_b
      R_bt[:, 0] = np.cross(u_b, a_b)
    else:
      R_bt[:, 0] = u_b
      R_bt[:, 1] = np.cross(a_b, u_b)
    T_bt = np.eye(4)
    T_bt[:3, :3] = R_bt
    T_bt[:3, 3] = tip_b - R_bt @ tip_tcp
    turn = np.degrees(np.arccos(np.clip((np.trace(T_ref_tool[:3, :3].T @ R_bt) - 1) / 2,
                                        -1, 1)))
    cands.append(dict(score=g["prio"] * 1000 + turn, T=T_bt, info=g, v=v, tilt=tilt,
                      depth=depth, open=open_m, low=low, off=off, turn=turn,
                      tip_b=tip_b, a_b=a_b, u_b=u_b))
  if not cands:
    return None, "; ".join(dict.fromkeys(reasons)) or "no grasp for this gripper"
  return min(cands, key=lambda c: c["score"]), ""


# ==============================================================================
# Gripper adapters
# ==============================================================================
class RobotiqSocket:
  """Robotiq 2F through the Robotiq URCap socket (port 63352), the same
  protocol as ur_rtde's robotiq_gripper.py. Works while RTDE control runs."""

  def __init__(self, ip, stroke_mm, port=63352, speed=64, force=100):
    self.s = socket.create_connection((ip, port), timeout=3.0)
    self.stroke, self.speed, self.force = stroke_mm, speed, force
    self.open_pos, self.closed_pos = 0, 230

  def _send(self, cmd):
    self.s.sendall((cmd + "\n").encode())
    return self.s.recv(1024).decode().strip()

  def get(self, var):
    r = self._send(f"GET {var}")
    return int(r.split()[1])

  def activate(self, confirm=True):
    if self.get("STA") == 3:
      return
    if confirm:
      input("[gripper] not active: activating opens and closes it fully. Enter = go ")
    self._send("SET ACT 1")
    t0 = time.time()
    while self.get("STA") != 3:
      if time.time() - t0 > 15:
        raise RuntimeError("Robotiq did not activate")
      time.sleep(0.2)

  def calibrate(self, confirm=True):
    if confirm:
      input("[gripper] calibration: opens fully, then closes fully (keep it empty). Enter ")
    self._goto_pos(0)
    self.open_pos = self.get("POS")
    self._goto_pos(255)
    self.closed_pos = self.get("POS")
    self._goto_pos(0)
    print(f"[gripper] open pos {self.open_pos}, closed pos {self.closed_pos}")

  def _goto_pos(self, pos, timeout=6.0):
    pos = int(np.clip(pos, 0, 255))
    self._send(f"SET POS {pos} SPE {self.speed} FOR {self.force} GTO 1")
    t0 = time.time()
    while self.get("PRE") != pos and time.time() - t0 < 1.0:
      time.sleep(0.02)
    while time.time() - t0 < timeout:
      obj = self.get("OBJ")
      if obj != 0:
        return obj
      time.sleep(0.05)
    return self.get("OBJ")

  def pos_of(self, width_mm):
    f = 1 - np.clip(width_mm / self.stroke, 0, 1)
    return self.open_pos + (self.closed_pos - self.open_pos) * f

  def width(self):
    p = self.get("POS")
    span = max(1, self.closed_pos - self.open_pos)
    return round(self.stroke * (1 - (p - self.open_pos) / span), 1)

  def open_to(self, width_mm):
    return self._goto_pos(self.pos_of(width_mm))

  def close(self):
    """Close with the set force. Returns (OBJ status, width mm):
    2 = stopped on an object while closing, 3 = closed fully (nothing)."""
    return self._goto_pos(255), self.width()

  def status(self):
    return self.get("OBJ"), self.width()

  def shutdown(self):
    with contextlib.suppress(Exception):
      self.s.close()


class ManualGripper:
  """No socket: you open/close it on the pendant when asked."""

  def activate(self, confirm=True):
    pass

  def calibrate(self, confirm=True):
    pass

  def open_to(self, width_mm):
    input(f"[gripper] OPEN the gripper to about {width_mm:.0f} mm (pendant), Enter ")
    return ""

  def close(self):
    input("[gripper] CLOSE the gripper now (pendant), Enter when closed ")
    return "", ""

  def status(self):
    return "", ""

  def shutdown(self):
    pass


class SimGripper(ManualGripper):
  def open_to(self, width_mm):
    return ""

  def close(self):
    return "", ""


# ==============================================================================
# Robot
# ==============================================================================
class ProtectiveStop(RuntimeError):
  pass


class GraspRobot:
  """ur_rtde UR5e with guarded moves and protective-stop recovery.
  UNTESTED ON HARDWARE."""
  moves_hardware = True

  def __init__(self, ip, a):
    import rtde_control
    import rtde_receive
    self.ip, self.a = ip, a
    self._rc, self._rr = rtde_control, rtde_receive
    self.c = rtde_control.RTDEControlInterface(ip)
    self.r = rtde_receive.RTDEReceiveInterface(ip)

  def get_tcp(self):
    return cal.from_ur_pose(self.r.getActualTCPPose())

  def _ask(self, label, kind):
    if self.a.confirm == "all" or (self.a.confirm == "grasp" and kind == "descend"):
      k = input(f"        Enter = move ({label}), q = stop: ").strip().lower()
      if k == "q":
        raise KeyboardInterrupt("stopped by operator")

  def _check_target(self, p):
    with contextlib.suppress(AttributeError):
      if not self.c.isPoseWithinSafetyLimits(p):
        raise ValueError("target outside the robot's safety limits")

  def stopped(self):
    with contextlib.suppress(Exception):
      return bool(self.r.isProtectiveStopped() or self.r.isEmergencyStopped())
    return False

  def move(self, T, label, speed=None, kind="move"):
    p = cal.to_ur_pose(T)
    print(f"[robot] {label}: moveL {np.round(p, 4).tolist()}")
    self._check_target(p)
    self._ask(label, kind)
    ok = self.c.moveL(p, speed or self.a.speed, self.a.acc)
    if not ok or self.stopped():
      if self.stopped():
        raise ProtectiveStop(f"protective stop during {label}")
      raise RuntimeError(f"moveL failed during {label}")

  def guarded(self, T, label, speed, limit_n):
    """Slow moveL that stops when the wrist force exceeds limit_n.
    Returns ('ok' | 'contact', max force N)."""
    p = cal.to_ur_pose(T)
    print(f"[robot] {label}: guarded moveL {np.round(p, 4).tolist()} (stop at {limit_n:g} N)")
    self._check_target(p)
    self._ask(label, "descend")
    with contextlib.suppress(Exception):
      self.c.zeroFtSensor()
    time.sleep(0.3)
    dist = np.linalg.norm(np.asarray(p[:3]) - np.asarray(self.r.getActualTCPPose()[:3]))
    self.c.moveL(p, speed, self.a.acc, True)
    t0, fmax = time.time(), 0.0
    timeout = dist / speed + 5.0
    time.sleep(0.1)
    while True:
      if self.stopped():
        raise ProtectiveStop(f"protective stop during {label}")
      f = float(np.linalg.norm(self.r.getActualTCPForce()[:3]))
      fmax = max(fmax, f)
      if f > limit_n:
        self.c.stopL(2.0)
        print(f"[robot] CONTACT {f:.1f} N during {label}: stopped")
        return "contact", fmax
      cur = np.asarray(self.r.getActualTCPPose()[:3])
      prog = self.c.getAsyncOperationProgress()
      if (prog < 0 and time.time() - t0 > 0.3) or np.linalg.norm(cur - p[:3]) < 3e-4:
        time.sleep(0.1)
        return "ok", fmax
      if time.time() - t0 > timeout:
        self.c.stopL(2.0)
        raise RuntimeError(f"timeout during {label}")
      time.sleep(0.01)

  def recover(self):
    """After a protective stop: you clear it, then reconnect."""
    print("\n[!] PROTECTIVE STOP / robot error. On the pendant: clear the stop "
          "(unlock), make sure nothing is pinched. Remove the part if it is in the gripper.")
    while True:
      input("    Press Enter when the robot is unlocked and in remote/ready mode ")
      try:
        with contextlib.suppress(Exception):
          self.c.disconnect()
        self.c = self._rc.RTDEControlInterface(self.ip)
        self.r = self._rr.RTDEReceiveInterface(self.ip)
        if not self.stopped():
          print("[+] reconnected")
          return
      except Exception as e:
        print(f"    reconnect failed: {e}")

  def close(self):
    with contextlib.suppress(Exception):
      self.c.stopScript()


class SimRobot:
  """Moves nothing (pose only). For --robot sim."""
  moves_hardware = False

  def __init__(self, T0):
    self.T = T0.copy()

  def get_tcp(self):
    return self.T.copy()

  def move(self, T, label, speed=None, kind="move"):
    self.T = T.copy()

  def guarded(self, T, label, speed, limit_n):
    self.T = T.copy()
    return "ok", 0.0

  def stopped(self):
    return False

  def recover(self):
    pass

  def close(self):
    pass


def translate(T, d):
  T2 = T.copy()
  T2[:3, 3] = T2[:3, 3] + d
  return T2


# ==============================================================================
# Plan
# ==============================================================================
def make_plan(objects, setups, reps, seed, color, methods=("vgap", "plain")):
  rng = random.Random(seed)
  objs = list(objects)
  rng.shuffle(objs)
  rows, k = [], 0
  for obj in objs:
    ss = list(setups)
    rng.shuffle(ss)
    for si, (place, yaw, tilt) in enumerate(ss):
      for rep in range(1, reps + 1):
        ms = list(methods)
        rng.shuffle(ms)
        for m in ms:
          k += 1
          rows.append(dict(trial_id=k, object=obj, color=color,
                           setup=f"{place}_yaw{yaw}_t{tilt:g}", placement=place,
                           yaw_label=yaw, tilt_deg=tilt, rep=rep, method=m))
  return rows


def parse_setups(s, obj):
  out = []
  for item in s.split(","):
    place, yaw, tilt = item.strip().split(":")
    if obj == "hemisphere":
      yaw = "0"     # yaw does not matter for the dome
    out.append((place, yaw, float(tilt)))
  return out


# ==============================================================================
# Report
# ==============================================================================
def report(paths, out_txt):
  rows = []
  for p in paths:
    with open(p) as f:
      rr = list(csv.DictReader(f))
    last = {}
    for r in rr:                     # newest row of each trial (--redo)
      last[r["trial_id"]] = r
    rows += list(last.values())
  lines = []
  P = lines.append
  P(f"V-GAP grasp trials: {len(rows)} trials from {len(paths)} file(s)")
  P("success = part lifted and still held after the 5 s hold (operator)")
  P("attempted = the robot went in (closed, collided, aborted, protective stop)\n")
  objs = [o for o in OBJECT_ORDER if any(r["object"] == o for r in rows)] + ["ALL"]
  for obj in objs:
    rs = [r for r in rows if obj == "ALL" or r["object"] == obj]
    P(f"== {obj} ==")
    for m in ("vgap", "plain"):
      ms = [r for r in rs if r["method"] == m]
      if not ms:
        continue
      n = len(ms)
      s = sum(r["outcome"] == "success" for r in ms)
      att = [r for r in ms if r["outcome"] in ATTEMPTED]
      fail_att = sum(r["outcome"] != "success" for r in att)
      lo, hi = ge.wilson(s, n)
      sa = sum(r["outcome"] == "success" for r in att)
      lo2, hi2 = ge.wilson(sa, len(att)) if att else (None, None)
      P(f"  {m:5s} success per trial {s}/{n} = {100*s/n:5.1f}% "
        f"[{100*lo:.0f}-{100*hi:.0f}]   per attempt {sa}/{len(att)}"
        + (f" = {100*sa/len(att):5.1f}% [{100*lo2:.0f}-{100*hi2:.0f}]" if att else "")
        + f"   failed attempts {fail_att}")
      cnt = {o: sum(r["outcome"] == o for r in ms) for o in OUTCOMES}
      P("        " + ", ".join(f"{o} {c}" for o, c in cnt.items() if c))
    # paired (same setup + rep): McNemar exact
    key = lambda r: (r["object"], r["setup"], r["rep"])
    v = {key(r): r["outcome"] == "success" for r in rs if r["method"] == "vgap"}
    p = {key(r): r["outcome"] == "success" for r in rs if r["method"] == "plain"}
    common = [k for k in v if k in p]
    if common:
      b = sum(v[k] and not p[k] for k in common)
      c = sum(p[k] and not v[k] for k in common)
      from math import comb
      nn = b + c
      pv = min(1.0, 2 * sum(comb(nn, i) for i in range(min(b, c) + 1)) / 2 ** nn) if nn else 1.0
      P(f"  paired {len(common)}: only V-GAP succeeded {b}, only plain {c}, "
        f"McNemar exact p = {pv:.3g}")
    pl = [r for r in rs if r["method"] == "plain" and r.get("plain_gate_would_pass") in ("True", "False")]
    for wp in ("True", "False"):
      g = [r for r in pl if r["plain_gate_would_pass"] == wp]
      if g:
        s = sum(r["outcome"] == "success" for r in g)
        P(f"  plain grasps the gate would {'ACCEPT' if wp == 'True' else 'REJECT'}: "
          f"{s}/{len(g)} succeeded")
    P("")
  txt = "\n".join(lines)
  if out_txt:
    with open(out_txt, "w") as f:
      f.write(txt + "\n")
  return txt


# ==============================================================================
# Main
# ==============================================================================
def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--mode", default="run", choices=["check-grasps", "setup", "tip-xy", "run", "report"])
  ap.add_argument("--report", nargs="+", default=["data/grasp_trials/*/trials.csv"])
  ap.add_argument("--gripper", choices=list(GRIPPERS), required=False)
  ap.add_argument("--gripper-io", default="robotiq", choices=["robotiq", "manual"],
                  help="robotiq = URCap socket 63352; manual = you use the pendant")
  ap.add_argument("--grip-speed", type=int, default=64, help="0-255")
  ap.add_argument("--grip-force", type=int, default=100, help="0-255")
  ap.add_argument("--pad-width", type=float, default=None)
  ap.add_argument("--finger-thickness", type=float, default=None)
  ap.add_argument("--allow-experimental", action="store_true",
                  help="allow the 2F-85 gearbox lip pinch")
  ap.add_argument("--setup-file", default="data/grasp_trials/setup.json")
  ap.add_argument("--tip-z", type=float, default=None,
                  help="fingertip end along TCP z, m (from --mode setup; default 0.110)")
  ap.add_argument("--tip-xy", default=None,
                  help="fingertip center x,y in the TCP frame, m (from --mode tip-xy; default 0,0)")
  ap.add_argument("--close-axis", choices=["x", "y"], default=None,
                  help="TCP axis the fingers close along (from --mode setup; default y)")
  ap.add_argument("--objects", default="l_block,t_block,gearbox,hemisphere")
  ap.add_argument("--color", default="white")
  ap.add_argument("--setups", default=SETUPS_DEFAULT,
                  help="placement:yaw:tilt, ... (yaw = how you turn the part from "
                       "your 0 mark; tilt = camera view, deg)")
  ap.add_argument("--reps", type=int, default=2, help="reps per setup and method")
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--resume", default=None, help="output folder of a run to continue")
  ap.add_argument("--vgap-bursts", type=int, default=2,
                  help="bursts at the view for V-GAP (first Valid is used; 2 = one "
                       "same-view retry, as in the pipeline)")
  ap.add_argument("--axis-ambig-deg", type=float, default=15.0, help="hemisphere")
  ap.add_argument("--robot", default="ur-rtde", choices=["ur-rtde", "sim"])
  ap.add_argument("--robot-ip", default="192.168.1.5")
  ap.add_argument("--serial", default=None)
  ap.add_argument("--hand-eye", default="../Python Files Disassembly/matrix_config/T_flange_cam_updated_zfix.npy")
  ap.add_argument("--side", default="+y", choices=list(tr.SIDES))
  ap.add_argument("--dist", type=float, default=0.40)
  ap.add_argument("--target", default=None, help="object center x,y,z m (skip finding)")
  ap.add_argument("--target-filter", action="store_true", help="(default now; kept for old commands)")
  ap.add_argument("--no-target-filter", action="store_true",
                  help="keep detections far from the part (default drops > --target-max-dist)")
  ap.add_argument("--target-max-dist", type=float, default=0.10)
  ap.add_argument("--capture-grip", default="keep", choices=["close", "keep"],
                  help="close the gripper before each capture so open fingers are not "
                       "seen (and detected) by the camera")
  ap.add_argument("--redo", default="", help="trial ids to run again with --resume, e.g. 11,12")
  ap.add_argument("--speed", type=float, default=0.08)
  ap.add_argument("--acc", type=float, default=0.2)
  ap.add_argument("--approach-speed", type=float, default=0.02)
  ap.add_argument("--lift-speed", type=float, default=0.03)
  ap.add_argument("--approach-dist", type=float, default=0.08, help="pre-grasp standoff, m")
  ap.add_argument("--hover", type=float, default=0.10, help="extra height above pre-grasp, m")
  ap.add_argument("--lift", type=float, default=0.10, help="lift height, m")
  ap.add_argument("--hold-s", type=float, default=5.0)
  ap.add_argument("--contact-force", type=float, default=25.0, help="N, guarded descent")
  ap.add_argument("--no-place-back", action="store_true")
  ap.add_argument("--confirm", default="all", choices=["all", "grasp", "none"])
  ap.add_argument("--table-z", type=float, default=0.008)
  ap.add_argument("--tip-clear", type=float, default=6.0, help="min fingertip-table gap, mm")
  ap.add_argument("--min-contact", type=float, default=8.0, help="min finger depth, mm")
  ap.add_argument("--finger-margin", type=float, default=2.0, help="finger-part gap, mm")
  ap.add_argument("--max-approach-tilt", type=float, default=35.0)
  ap.add_argument("--max-grasp-offset", type=float, default=0.06,
                  help="grasp must be within this of the detected object, m")
  ap.add_argument("--max-tilt", type=float, default=45.0)
  ap.add_argument("--max-reach", type=float, default=0.85)
  ap.add_argument("--min-base-dist", type=float, default=0.20)
  ap.add_argument("--min-tcp-height", type=float, default=0.08)
  ap.add_argument("--min-cam-height", type=float, default=0.08)
  ap.add_argument("--min-tip-height", type=float, default=0.03)
  ap.add_argument("--min-tip-object", type=float, default=0.10)
  ap.add_argument("--no-gui", action="store_true")
  ap.add_argument("--methods", default="vgap,plain",
                  help="vgap,plain (paired trials) or one of them (demos)")
  ap.add_argument("--demo-video", nargs="?", const="auto", default=None,
                  help="record the split-screen demo video (live RGB + YOLO, 3D check, 2D gate, "
                       "verdict) of the run; default <out>/demo_video.mp4. Demo runs only.")
  ap.add_argument("--demo-fps", type=float, default=12.0)
  ap.add_argument("--demo-size", default="1920x1080")
  ap.add_argument("--demo-no-live-yolo", action="store_true")
  ap.add_argument("--out", default=None)
  a = ap.parse_args()

  if a.mode == "report":
    paths = [p for pat in a.report for p in glob.glob(pat)]
    if not paths:
      raise SystemExit("no trials.csv found")
    out = os.path.join(os.path.dirname(paths[0]), "..", "grasp_report.txt") \
        if len(paths) > 1 else os.path.join(os.path.dirname(paths[0]), "grasp_report.txt")
    print(report(paths, os.path.normpath(out)))
    return

  if not a.gripper:
    raise SystemExit("--gripper 2f85 or 2f140 is needed (see the gripper label)")
  gp = dict(GRIPPERS[a.gripper])
  if a.pad_width:
    gp["pad_w"] = a.pad_width
  if a.finger_thickness:
    gp["finger_t"] = a.finger_thickness

  if a.mode == "check-grasps":
    check_grasps(a.gripper, gp, a.allow_experimental)
    return

  setup = {}
  if os.path.exists(a.setup_file):
    with open(a.setup_file) as f:
      setup = json.load(f)
  if a.mode == "setup":
    return run_setup(a, gp)
  if a.mode == "run" and a.robot != "sim" and a.tip_z is None and "tip_z" not in setup:
    raise SystemExit(f"no fingertip offset: run --mode setup first (writes {a.setup_file}) "
                     "or pass --tip-z")
  a.tip_z = a.tip_z if a.tip_z is not None else setup.get("tip_z", 0.110)
  a.close_axis = a.close_axis or setup.get("close_axis", "y")
  a.tip_xy = cal.parse_list(a.tip_xy) if a.tip_xy else setup.get("tip_xy", [0.0, 0.0])
  if a.mode == "tip-xy":
    return run_tip_xy(a, gp)
  if a.mode == "run" and a.robot != "sim" and "tip_xy" not in setup and a.tip_xy == [0.0, 0.0]:
    print("[!] fingertip x/y offset never measured (--mode tip-xy); assuming the fingertips "
          "are on the TCP axis")

  return run_trials(a, gp, setup)


def run_setup(a, gp):
  """Measure the fingertip offset (touch the table) and the closing axis.
  Step 1 needs freedrive (pendant in LOCAL mode); step 2 moves the arm over
  RTDE (pendant in REMOTE mode). The offset is saved right after step 1, and
  --tip-z skips step 1 (e.g. --tip-z 0.131 from an earlier run)."""
  import rtde_control
  import rtde_receive
  grip = RobotiqSocket(a.robot_ip, gp["stroke"], speed=a.grip_speed, force=a.grip_force) \
      if a.gripper_io == "robotiq" else ManualGripper()
  grip.activate()
  r = rtde_receive.RTDEReceiveInterface(a.robot_ip)
  os.makedirs(os.path.dirname(a.setup_file) or ".", exist_ok=True)
  res = dict(gripper=a.gripper, table_z=a.table_z)
  if os.path.exists(a.setup_file):
    with open(a.setup_file) as f:
      res.update(json.load(f))

  def save():
    res["measured"] = datetime.now().isoformat(timespec="seconds")
    with open(a.setup_file, "w") as f:
      json.dump(res, f, indent=2)

  if a.tip_z is not None:
    res["tip_z"] = round(float(a.tip_z), 4)
    print(f"Step 1 skipped: tip_z {res['tip_z']} m (--tip-z)")
  else:
    print("\nStep 1, fingertip offset. Pendant in LOCAL mode; tool pointing straight down.")
    grip.open_to(50.0)
    input("  With freedrive, lower the gripper until the fingertips JUST touch the table "
          "(gripper ~50 mm open). Release freedrive, then press Enter ")
    T = cal.from_ur_pose(r.getActualTCPPose())
    tilt = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))
    tip_z = T[2, 3] - a.table_z
    print(f"  TCP z {T[2, 3]*1000:.1f} mm, tool tilt {tilt:.1f} deg from straight down")
    if tilt > 5:
      print("  [!] tool is not straight down; tip offset below is approximate")
    print(f"  fingertip offset tip_z = TCP z - table z ({a.table_z*1000:.1f} mm) = "
          f"{tip_z*1000:.1f} mm  (with table at 4.4 mm it would be {(T[2,3]-0.0044)*1000:.1f})")
    res.update(tip_z=round(float(tip_z), 4),
               tcp_at_touch=[round(v, 5) for v in cal.to_ur_pose(T)],
               tool_tilt_deg=round(float(tilt), 2))
    save()
    print(f"  [+] saved tip_z to {a.setup_file}")

  input("\nStep 2, closing axis. With freedrive raise the gripper ~5 cm (clear of everything), "
        "then switch the pendant to REMOTE control and press Enter ")
  while True:
    try:
      c = rtde_control.RTDEControlInterface(a.robot_ip)
      break
    except RuntimeError as e:
      input(f"  {e}\n  Switch the pendant to Remote Control (top right), then Enter ")
  T0 = cal.from_ur_pose(r.getActualTCPPose())
  T1 = T0.copy()
  T1[:3, 3] += T0[:3, 0] * 0.02
  input("  The TCP will move 20 mm along the TOOL x axis and back. Enter = move ")
  c.moveL(cal.to_ur_pose(T1), 0.02, 0.1)
  k = input("  Did it move in the direction the fingers open/close (toward one finger)? y/n: ")
  c.moveL(cal.to_ur_pose(T0), 0.02, 0.1)
  res["close_axis"] = "x" if k.strip().lower().startswith("y") else "y"
  save()
  print(f"\n[+] saved {a.setup_file}: tip_z {res['tip_z']} m, fingers close along tool "
        f"{res['close_axis']}")
  c.stopScript()
  grip.shutdown()


def run_tip_xy(a, gp):
  """Fingertip center x/y in the TCP frame: touch the same dot twice with the
  wrist turned ~180 deg between touches (pendant LOCAL mode, freedrive)."""
  import rtde_receive
  grip = RobotiqSocket(a.robot_ip, gp["stroke"], speed=a.grip_speed, force=a.grip_force) \
      if a.gripper_io == "robotiq" else ManualGripper()
  grip.activate()
  r = rtde_receive.RTDEReceiveInterface(a.robot_ip)
  print("Fingertip x/y offset. Draw a small dot on the table (or on tape).")
  grip.close()
  Ts = []
  for k, msg in enumerate([
      "Tool straight down. With freedrive, put the CENTER of the closed fingertips right on "
      "the dot (just touching). Enter ",
      "Raise a few cm, turn wrist 3 by about 180 deg (Move tab), keep the tool straight down, "
      "then freedrive back so the fingertip center is on the SAME dot. Enter "]):
    input(f"  {k + 1}. {msg}")
    T = cal.from_ur_pose(r.getActualTCPPose())
    print(f"     TCP {np.round(np.array(cal.to_ur_pose(T)[:3]) * 1000, 1)} mm")
    Ts.append(T)
  (R1, t1), (R2, t2) = [(T[:3, :3], T[:3, 3]) for T in Ts]
  ang = np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
  if ang < 90:
    raise SystemExit(f"wrist only turned {ang:.0f} deg between touches; turn ~180 and redo")
  D = R1 - R2
  rhs = t2 - t1 - D[:, 2] * a.tip_z
  xy, *_ = np.linalg.lstsq(D[:, :2], rhs, rcond=None)
  resid = np.linalg.norm(D[:, :2] @ xy - rhs) * 1000
  print(f"\n  wrist turned {ang:.0f} deg; fingertip center in TCP frame: x {xy[0]*1000:.1f} mm, "
        f"y {xy[1]*1000:.1f} mm (fit residual {resid:.1f} mm)")
  res = {}
  if os.path.exists(a.setup_file):
    with open(a.setup_file) as f:
      res = json.load(f)
  res.update(tip_xy=[round(float(v), 4) for v in xy], tip_xy_turn_deg=round(float(ang), 1),
             tip_xy_tcps=[[round(v, 5) for v in cal.to_ur_pose(T)] for T in Ts],
             tip_xy_measured=datetime.now().isoformat(timespec="seconds"))
  os.makedirs(os.path.dirname(a.setup_file) or ".", exist_ok=True)
  with open(a.setup_file, "w") as f:
    json.dump(res, f, indent=2)
  print(f"  [+] saved to {a.setup_file}")
  grip.shutdown()


def run_trials(a, gp, setup):
  objects = [o.strip() for o in a.objects.split(",") if o.strip()]
  usable = []
  for o in objects:
    if grasp_variants(o, a.gripper, a.allow_experimental):
      usable.append(o)
    else:
      print(f"[!] {o}: no grasp for the {a.gripper} -> left out")
  if not usable:
    raise SystemExit("nothing to grasp")

  # ---- output / plan
  if a.resume:
    out_dir = a.resume
    with open(os.path.join(out_dir, "plan.csv")) as f:
      plan = list(csv.DictReader(f))
    done = set()
    if os.path.exists(os.path.join(out_dir, "trials.csv")):
      with open(os.path.join(out_dir, "trials.csv")) as f:
        done = {r["trial_id"] for r in csv.DictReader(f)}
    redo = {x.strip() for x in a.redo.split(",") if x.strip()}
    done -= redo
    if redo:
      print(f"[+] redoing trials {sorted(redo, key=int)} (the old rows stay in trials.csv; "
            "the report uses the newest row of each trial)")
    for r in plan:
      r["tilt_deg"] = float(r["tilt_deg"])
    print(f"[+] resuming {out_dir}: {len(done)}/{len(plan)} trials done")
  else:
    seed = a.seed if a.seed is not None else int(time.time()) % 100000
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = a.out or os.path.join("data", "grasp_trials", f"{a.gripper}_{a.color}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    plan = []
    for o in usable:
      plan += [r for r in make_plan([o], parse_setups(a.setups, o), a.reps, seed, a.color,
                                    [m.strip() for m in a.methods.split(",") if m.strip()])]
    # objects in random order, ids renumbered
    rng = random.Random(seed)
    order = list(usable)
    rng.shuffle(order)
    plan = sorted(plan, key=lambda r: (order.index(r["object"]), r["trial_id"]))
    for i, r in enumerate(plan, 1):
      r["trial_id"] = str(i)
    with open(os.path.join(out_dir, "plan.csv"), "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(plan[0]))
      w.writeheader()
      w.writerows(plan)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
      json.dump(dict(vars(a), seed=seed, gripper_params=gp, setup=setup,
                     success_rule="lifted and still held after the hold (operator y/n)"),
                f, indent=2, default=str)
    done = set()
    print(f"[+] plan: {len(plan)} trials (seed {seed}) -> {out_dir}/plan.csv")

  tfile = os.path.join(out_dir, "trials.csv")
  new = not os.path.exists(tfile)
  tf = open(tfile, "a", newline="")
  tw = csv.DictWriter(tf, fieldnames=FIELDS, extrasaction="ignore")
  if new:
    tw.writeheader()

  # ---- hardware
  if a.robot == "sim":
    import sim_stress_test as st
    for k, rel in st.MESHES.items():
      vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(st.ROOT, rel)
      vp.OBJECT_CONFIGS[k]["cad_units"] = "mm"
    X = np.load(a.hand_eye) if os.path.exists(a.hand_eye) else None
    if X is None:
      X = np.eye(4)
      X[:3, 3] = [-0.031, -0.034, -0.003]
    robot = SimRobot(cal.from_ur_pose(tr.OBSERVE_POSE))
    grip = SimGripper()
    gui = False
  else:
    X = np.load(a.hand_eye).astype(float)
    robot = GraspRobot(a.robot_ip, a)
    grip = RobotiqSocket(a.robot_ip, gp["stroke"], speed=a.grip_speed, force=a.grip_force) \
        if a.gripper_io == "robotiq" else ManualGripper()
    grip.activate()
    if a.gripper_io == "robotiq":
      grip.calibrate()
    gui = not a.no_gui
  a.tool_tip = np.array([a.tip_xy[0], a.tip_xy[1], a.tip_z])
  print(f"[+] fingertip center in TCP frame ({a.tip_xy[0]*1000:.1f}, {a.tip_xy[1]*1000:.1f}, "
        f"{a.tip_z*1000:.0f}) mm, fingers close along tool "
        f"{a.close_axis}, table z {a.table_z*1000:.1f} mm")

  vp.OBJECT_CONFIGS["hemisphere"]["axis_ambig_deg"] = a.axis_ambig_deg
  cad_cache = {}
  for k in OBJECT_ORDER:
    with contextlib.redirect_stdout(io.StringIO()):
      cad_cache[k] = vp.load_cad_model(vp.OBJECT_CONFIGS[k])
  P_obj = {k: cad_points(k, 20000) for k in OBJECT_ORDER}

  cams = {}

  def camera(obj):
    if a.robot == "sim":
      if obj not in cams:
        cams.clear()
        cams[obj] = cal.SimCamera(obj, a.color, 0, X)
        if a.demo_video and hasattr(cams[obj], "_frames") and "rec" not in demo:
          start_demo(cams[obj])          # tests only (a sim camera with _frames)
      return cams[obj]
    if "real" not in cams:
      cams["real"] = cal.RealSenseCamera(a.serial)
      cams["real"].gripper_roi = tuple(vp.GRIPPER_ROI) if vp.GRIPPER_ROI else None
      cams["real"].max_dist = 0 if a.no_target_filter else a.target_max_dist
      cams["real"].X = X
      if a.demo_video:
        start_demo(cams["real"])
    return cams["real"]

  demo = {}

  def start_demo(cam):
    import demo_recorder
    vw, vh = (int(v) for v in a.demo_size.lower().split("x"))
    rec = demo_recorder.DemoRecorder(
        os.path.join(out_dir, "demo_video.mp4") if a.demo_video == "auto" else a.demo_video,
        cam, None, "", fps=a.demo_fps, size=(vw, vh), live_yolo=not a.demo_no_live_yolo)
    rec.conflict_text = "NO grasp"
    demo["rec"] = rec
    # arm moves show in the status line
    for name in ("move", "guarded"):
      orig = getattr(robot, name)

      def wrapped(T, label, *args, _orig=orig, **kw):
        rec.set_status(f"arm: {label}")
        try:
          return _orig(T, label, *args, **kw)
        finally:
          rec.set_status("")
      setattr(robot, name, wrapped)

  def dstage(text):
    if "rec" in demo:
      demo["rec"].set_stage(text)

  def dstatus(text):
    if "rec" in demo:
      demo["rec"].set_status(text)

  center_of = {}
  T_down = cal.from_ur_pose(tr.OBSERVE_POSE)

  def locate(obj, setup_key, cam):
    if a.target:
      return np.asarray(cal.parse_list(a.target))
    if setup_key in center_of:
      return center_of[setup_key]
    if a.robot == "sim":
      c = cam.object_center()
    else:
      if a.capture_grip == "close":
        grip.close()
      robot.move(T_down, "observe pose (find the part)")
      obs = None
      while obs is None:
        for _ in range(3):
          obs = cam.capture(robot.get_tcp())
          if obs is not None:
            break
        if obs is None:
          k = input("  [-] YOLO did not find the part from the observe pose. Check it is in "
                    "view. Enter = try again, s = skip this trial (redo with --resume): ")
          if k.strip().lower() == "s":
            return None
      print(f"  found: YOLO {obs.object_key} ({obs.confidence:.2f}) "
            f"{getattr(cam, 'last_filter', {}).get('dropped', '')}")
      half_h = min(vp.OBJECT_CONFIGS[obj]["expected_extent_mm"]) / 2000.0
      c = tr.locate(obs, robot.get_tcp() @ X, half_h)
    center_of[setup_key] = c
    return c

  def sim_hold(obj, cam, cand):
    """Sim only: would the fingers close on the TRUE part without a collision?"""
    T = cam.T_base_obj
    P = P_obj[obj] @ T[:3, :3].T + T[:3, 3]
    hit = any(points_in_box(P, b).any() for b in
              finger_boxes(cand["tip_b"], cand["a_b"], cand["u_b"], cand["open"], gp))
    n, wid, dep, off = closing_contact(P, cand["tip_b"], cand["a_b"], cand["u_b"],
                                       cand["open"], gp)
    return (not hit) and n > 50 and dep > 0.005 and wid < cand["open"] - 0.002

  def draw_preview(obs, out_vis, T_bt_cap, cand, path, title):
    img = obs.color.copy()
    for m, col in ((obs.mask, (0, 255, 0)), (out_vis, (0, 0, 255))):
      if m is None:
        continue
      cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
      cv2.drawContours(img, cs, -1, col, 2)
    if cand is not None:
      T_cb = np.linalg.inv(T_bt_cap @ X)
      for c, R, h in finger_boxes(cand["tip_b"], cand["a_b"], cand["u_b"], cand["open"], gp):
        corners = np.array([c + R @ (np.array([i, j, k]) * h) for i in (-1, 1)
                            for j in (-1, 1) for k in (-1, 1)])
        pc = corners @ T_cb[:3, :3].T + T_cb[:3, 3]
        if np.all(pc[:, 2] > 0.01):
          uv = (pc[:, :2] / pc[:, 2:3]) @ obs.K[:2, :2].T + obs.K[:2, 2]
          hull = cv2.convexHull(uv.astype(np.int32))
          cv2.polylines(img, [hull], True, (255, 128, 0), 2)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imwrite(path, img)
    if gui:
      cv2.imshow("grasp preview (green mask, red CAD, blue fingers)", img)
      cv2.waitKey(300)

  def finish(row):
    tw.writerow(row)
    tf.flush()
    if "rec" in demo:
      demo["rec"].event("trial", trial=row["trial_id"], method=row["method"],
                        outcome=row["outcome"])
      demo["rec"].set_status(f"RESULT: {row['outcome'].replace('_', ' ')}")

  try:
    for prow in plan:
      if prow["trial_id"] in done:
        continue
      obj, method = prow["object"], prow["method"]
      skey = (obj, prow["setup"])
      row = {k: "" for k in FIELDS}
      row.update({k: prow[k] for k in prow if k in FIELDS})
      row["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
      t_start = time.time()
      tdir = os.path.join(out_dir, "trials", f"t{int(prow['trial_id']):03d}_{obj}_{method}")
      os.makedirs(tdir, exist_ok=True)
      row["trial_dir"] = tdir
      print("\n" + "=" * 78)
      print(f"Trial {prow['trial_id']}/{len(plan)}: {obj} {a.color}, {prow['placement']}, "
            f"yaw {prow['yaw_label']} deg, view {prow['tilt_deg']:g} deg, rep {prow['rep']}, "
            f"method {method.upper()}")
      cam = camera(obj)
      if "rec" in demo:
        demo["rec"].live_paused = ""
        cfg = vp.OBJECT_CONFIGS[obj]
        demo["rec"].object_key, demo["rec"].tau_c = obj, cfg["tau_c"]
        demo["rec"].f3_min = cfg.get("f3_min") if vp.CHECK_3D else None
      tag = "V-GAP grasp" if method == "vgap" else "Plain RANSAC+ICP grasp"
      dstage(f"{tag}: {obj.replace('_block', '')} {prow['placement']}, "
             f"{prow['tilt_deg']:g} deg view")
      try:
        if a.robot != "sim":
          input(f"  Put the {obj} {ge.PLACEMENTS[prow['placement']]}, turned "
                f"{prow['yaw_label']} deg from your 0 mark, on the taped spot. Enter ")
        center = locate(obj, skey, cam)
        if center is None:
          print("  skipped (not logged); --resume will redo it")
          continue
        if a.robot != "sim":
          cam.target = None if a.no_target_filter else center
        T_view = tr.view_pose(center, prow["tilt_deg"], a.side, a.dist, X)
        tr.check_view(T_view, X, center, a)
        row["view_tcp"] = json.dumps([round(v, 5) for v in cal.to_ur_pose(T_view)])
        if a.capture_grip == "close":
          grip.close()
        robot.move(T_view, f"view {prow['tilt_deg']:g} deg")
        T_cap = robot.get_tcp()

        # ---- perception
        obs = out = None
        nb = a.vgap_bursts if method == "vgap" else 1
        for b in range(max(nb, 1) + (2 if method == "plain" else 0)):
          ge._BASELINE.clear()
          o = cam.capture(T_cap)
          if o is None:
            continue
          obs = o
          out = cal.run_pipeline(obs, cad_cache, True)
          Tb = ge.baseline_pose(obs, cad_cache)
          if Tb is not None:
            out["baseline_T_c2c"] = Tb
          row["bursts"] = b + 1
          if "rec" in demo:
            if method == "vgap":
              demo["rec"].on_burst(obs, out, cad_cache.get(out.get("final_object", obs.object_key)))
            elif Tb is not None:
              kp = vp.OBJECT_CONFIGS[obs.object_key]
              IBp, IMp, Cp = cal.score_pose(obs, cad_cache[obs.object_key], Tb, kp)
              Mp = vp.project_silhouette(cad_cache[obs.object_key].dense_points, Tb, obs.K,
                                         obs.mask.shape)[0]
              demo["rec"].on_burst(obs, dict(
                  state="plain", reason="", C=Cp, IB=IBp, IM=IMp, Mproj=Mp,
                  F3=vp.scan_fit_3d(out.get("scan"), cad_cache[obs.object_key], Tb),
                  T_cad_to_cam=Tb, scan=out.get("scan"), final_object=obs.object_key),
                  cad_cache[obs.object_key])
          if method == "plain" or out["state"] == "Valid" or b + 1 >= nb:
            break
        if obs is None:
          row.update(outcome="no_detection")
          finish(row)
          continue
        fin = out.get("final_object", obs.object_key)
        Tb_ = out.get("baseline_T_c2c")
        print(f"  YOLO {obs.object_key} -> {fin}: V-GAP {out.get('state')} "
              f"{out.get('reason', '')} C={out.get('C', 0):.3f} F3={out.get('F3', 0):.2f}"
              + ("" if Tb_ is None else "; plain pose C=%.3f" % cal.score_pose(
                  obs, cad_cache[obs.object_key], Tb_, vp.OBJECT_CONFIGS[obs.object_key])[2]))
        row.update(yolo_object=obs.object_key, final_object=fin,
                   class_swapped=out.get("class_swapped", False),
                   gate_state=out.get("state"), gate_reason=out.get("reason", ""),
                   C=round(out.get("C", 0), 4), F3=round(out.get("F3", 0), 4),
                   margin=round(out.get("margin", float("nan")), 4),
                   runtime_s=round(out.get("runtime_s", float("nan")), 2))
        Tb = out.get("baseline_T_c2c")
        if Tb is not None:
          prm = vp.OBJECT_CONFIGS[obs.object_key]
          _, _, Cb = cal.score_pose(obs, cad_cache[obs.object_key], Tb, prm)
          row["plain_C"] = round(Cb, 4)
          if fin == obs.object_key and out.get("scan") is not None:
            F3b = vp.scan_fit_3d(out["scan"], cad_cache[fin], Tb)
            row["plain_F3"] = round(F3b, 4)
            row["plain_gate_would_pass"] = bool(Cb >= prm["tau_c"] and
                                                F3b >= prm.get("f3_min", 0))
          if "T_cad_to_cam" in out and fin == obs.object_key:
            er, et = cal.pose_error(out["T_cad_to_cam"], Tb, cad_cache[fin])
            row.update(plain_vs_vgap_rot_deg=round(er, 2), plain_vs_vgap_trans_mm=round(et, 2))

        if method == "vgap":
          if out.get("state") != "Valid":
            row.update(outcome="no_grasp_gate", pose_used="none")
            draw_preview(obs, out.get("Mproj"), T_cap, None, os.path.join(tdir, "preview.png"),
                         f"V-GAP {out.get('state')} {out.get('reason', '')}: no grasp")
            finish(row)
            continue
          T_c2c, gobj = out["T_cad_to_cam"], fin
          row["pose_used"] = "vgap"
        else:
          if Tb is None:
            row.update(outcome="no_grasp_unsafe", pose_used="none",
                       refused_reason="no RANSAC+ICP pose")
            finish(row)
            continue
          T_c2c, gobj = Tb, obs.object_key
          row["pose_used"] = "plain"
        T_bo = T_cap @ X @ T_c2c
        row.update(tr.pose_fields(T_bo, "obj"))

        cand, why = plan_grasp(gobj, T_bo, T_down, gp, a, P_obj[gobj], center)
        vis = vp.project_silhouette(cad_cache[gobj].dense_points, T_c2c, obs.K,
                                    obs.mask.shape)[0]
        draw_preview(obs, vis, T_cap, cand, os.path.join(tdir, "preview.png"),
                     f"{method} {gobj} C={row['C'] if method == 'vgap' else row['plain_C']} "
                     + (cand["info"]["name"] if cand else f"REFUSED: {why[:60]}"))
        with contextlib.redirect_stdout(io.StringIO()):
          obs.meta["timestamp"] = f"t{prow['trial_id']}_{obs.meta.get('timestamp', '')}"
          if a.robot != "sim":
            vp.save_trial(obs, out, root=tdir)
        if cand is None:
          print(f"  [-] no safe grasp: {why}")
          row.update(outcome="no_grasp_unsafe", refused_reason=why)
          finish(row)
          continue
        row.update(grasp_name=cand["info"]["name"], grasp_mirrored=cand["v"]["mirrored"],
                   approach_tilt_deg=round(cand["tilt"], 1),
                   contact_depth_mm=round(cand["depth"] * 1000, 1),
                   open_mm=round(cand["open"] * 1000, 1),
                   expected_width_mm=cand["info"]["width"],
                   tip_above_table_mm=round((cand["low"] - a.table_z) * 1000, 1),
                   grasp_offset_mm=round(cand["off"] * 1000, 1),
                   grasp_tcp=json.dumps([round(v, 5) for v in cal.to_ur_pose(cand["T"])]))
        print(f"  grasp: {cand['info']['name']}, approach tilt {cand['tilt']:.0f} deg, "
              f"finger depth {cand['depth']*1000:.0f} mm, open {cand['open']*1000:.0f} mm, "
              f"tips {row['tip_above_table_mm']} mm above table")
        if a.robot == "sim":
          row["sim_would_hold"] = sim_hold(obj, cam, cand)

        # ---- motion
        if "rec" in demo:
          demo["rec"].live_paused = "pose locked: live detection paused while grasping"
        a_b = cand["a_b"]
        T_g = cand["T"]
        T_pre = translate(T_g, -a_b * a.approach_dist)
        T_hov = translate(T_pre, np.array([0, 0, a.hover]))
        grip.open_to(cand["open"] * 1000)
        robot.move(T_hov, "hover above grasp")
        robot.move(T_pre, "pre-grasp")
        res, fmax = robot.guarded(T_g, "DESCEND to grasp", a.approach_speed, a.contact_force)
        row["max_force_N"] = round(fmax, 1)
        if res == "contact":
          row.update(outcome="fail_collision", note="contact on the way down")
          robot.move(T_pre, "back off", speed=a.approach_speed)
          robot.move(T_hov, "hover")
          finish(row)
          continue
        dstatus("closing the gripper")
        st_c, w_c = grip.close()
        row.update(obj_status_close=st_c, width_close_mm=w_c)
        if st_c == 3:
          print("  [-] gripper closed on nothing")
          row.update(outcome="fail_miss", held_5s=False)
          grip.open_to(cand["open"] * 1000)
          robot.move(T_pre, "retreat", speed=a.lift_speed)
          robot.move(T_hov, "hover")
          finish(row)
          continue
        T_lift = translate(T_g, np.array([0, 0, a.lift]))
        robot.move(T_lift, "LIFT", speed=a.lift_speed)
        print(f"  holding {a.hold_s:g} s ...")
        dstatus(f"holding {a.hold_s:g} s")
        time.sleep(a.hold_s if a.robot != "sim" else 0)
        st_h, w_h = grip.status()
        row.update(obj_status_hold=st_h, width_hold_mm=w_h)
        if a.robot == "sim":
          held, shifted = bool(row["sim_would_hold"]), False
        else:
          k = ""
          while k not in ("y", "n", "s"):
            k = input("  Part still held off the table after the hold? "
                      "y = held, s = held but shifted/rotated in the fingers, n = not held: "
                      ).strip().lower()
          held, shifted = k in ("y", "s"), k == "s"
          if st_h == 3 and held:
            print("  [!] Robotiq says no object; logged both")
        row.update(held_5s=held, shifted=shifted,
                   outcome="success" if held else "fail_dropped")
        if held and not a.no_place_back:
          robot.move(translate(T_g, np.array([0, 0, 0.02])), "place back (above)",
                     speed=a.lift_speed)
          robot.guarded(translate(T_g, np.array([0, 0, 0.002])), "place back down",
                        a.approach_speed, a.contact_force)
        grip.open_to(cand["open"] * 1000)
        robot.move(T_pre if held else T_lift, "retreat", speed=a.lift_speed)
        robot.move(T_hov, "hover")
        if a.robot != "sim":
          note = input("  Note (Enter = none): ").strip()
          row["note"] = note
        finish(row)
      except KeyboardInterrupt:
        k = input("\n  Stopped. a = abort this trial for SAFETY (counts as failure), "
                  "s = skip without logging (redo later with --resume), x = exit: ").strip()
        if k == "a":
          row.update(outcome="aborted_safety", note="operator abort (safety)")
          finish(row)
        if k == "x":
          break
        with contextlib.suppress(Exception):
          grip.open_to(80)
      except ProtectiveStop as e:
        print(f"  [!] {e}")
        row.update(outcome="protective_stop", note=str(e))
        finish(row)
        robot.recover()
      except ValueError as e:
        print(f"  [-] view/target refused: {e}")
        row.update(outcome="no_grasp_unsafe", refused_reason=f"view: {e}")
        finish(row)
      except RuntimeError as e:
        print(f"  [!] {e}")
        if robot.stopped():
          row.update(outcome="protective_stop", note=str(e))
          finish(row)
          robot.recover()
        else:
          k = input("  Robot error. s = skip (redo with --resume), x = exit: ").strip()
          if k == "x":
            break
      finally:
        row["trial_s"] = round(time.time() - t_start, 1)
  finally:
    tf.close()
    with contextlib.suppress(Exception):
      robot.move(T_down, "observe pose (end)")
    robot.close()
    grip.shutdown()
    if "rec" in demo:
      with contextlib.suppress(Exception):
        demo["rec"].close()
    for c in cams.values():
      with contextlib.suppress(Exception):
        c.close()
  print(report([tfile], os.path.join(out_dir, "grasp_report.txt")))


if __name__ == "__main__":
  main()
