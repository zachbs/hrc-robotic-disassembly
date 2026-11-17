"""V-GAP trial runner on the UR5e: semester-demo style views, 10 bursts each.

For ONE object and color per run:
  1. The robot goes to OBSERVE_POSE, finds the object you put down (YOLO
     mask + depth), and moves the camera straight above it.
  2. For each view angle (default 0, 30, 45, 60 deg from straight down) it
     tilts the camera in one vertical plane, the same kind of move as the
     45 deg orbit in semester_demo.py: the tool only rotates about the base
     x axis, so the wrist never winds around the object.
  3. At each angle it asks you to put the object in orientation 1..4 (turn it
     90 deg about its vertical axis each time, same spot; the hemisphere has
     only 1). Press SPACE in the preview window (or Enter) when it is ready.
  4. Per orientation it takes 10 bursts and runs the full pipeline on each
     (YOLO -> RANSAC -> ICP -> CAD projection -> C gate). Every result is
     saved, pass or fail.
  5. It shows you one result per distinct pose and asks y/n: does the CAD sit
     on the object? The camera does not move during the 10 bursts, so the
     other repeats are labelled by how far they are from that pose (correct
     <= 5 deg / 5 mm, wrong > 10 deg / 10 mm). No hand-eye accuracy needed.
  6. Around the confirmed pose it also scores deliberately wrong poses
     (injected errors, as in calibrate_tau_c_ur5e.py) so tau_c has wrong
     examples to separate from.
  7. Before the next angle it goes back above the object and finds it again,
     in case it moved while you turned it.

Outputs in --out (default data/trials/<object>_<color>_<time>/):
  captures.csv        one row per burst: gate state, C, IB, IM, margin, UD,
                      U_live, ICP fitness / RMSE, 3D overlap, fallback,
                      class check, runtime, pose (camera and base frame),
                      label, deviation from the view's median pose
  calib_<object>_<color>.csv  rows in the format recommend_tau_c.py reads
  summary_by_view.csv / summary.txt  means, std and pass rates per
                      object x angle x orientation, per angle, per object
  trials/             RGB, depth, mask and result.json of every burst
Then, after all objects:
  python recommend_tau_c.py data/trials/*/calib_*.csv --sweep-weights
gives the best tau_c per object and the best wB / wM weights.

Motion safety: every move is a slow moveL (0.1 m/s like the demo's orbit)
and waits for Enter unless --no-confirm. A view is skipped (with the reason
printed) if the camera, gripper tip or TCP would come within the set
clearances of the table, the object or the base, or out of reach. 90 deg
(camera level with the object) and even 75 put the wrist on the table at
0.4 m, so the default top angle is 60. Run --plan-only first and check the poses on the pendant.
UNTESTED ON HARDWARE.

Example (from the folder with vgap_pipeline.py, the YOLO .pt and cad/):
  python vgap_trials_ur5e.py --object l_block --color white --robot ur-rtde \
      --robot-ip 192.168.1.5 --serial 337122075128 \
      --hand-eye matrix_config/T_flange_cam.npy --tool-tip 0,0,0.086
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
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vgap_pipeline as vp  # noqa: E402
import calibrate_tau_c_ur5e as cal  # noqa: E402  (robot / camera adapters)

OBSERVE_POSE = [0.200230894137112, -0.509647957735381, 0.5, 3.14159, 0.0, 0.0]
ORIENTATIONS = dict(gearbox=4, l_block=4, t_block=4, hemisphere=1)
SIDES = {"+y": (0, 1), "-y": (0, -1), "+x": (1, 0), "-x": (-1, 0)}


# ==============================================================================
# Views in one vertical plane (semester_demo orbit)
# ==============================================================================
def tool_rotation(tilt_deg, side):
  """Tool orientation for a tilt: straight down rotated about one base axis.

  side "+y" gives exactly the demo's family: rotvec (pi - tilt, 0, 0), i.e.
  [.., 2.356, 0, 0] at 45 deg, camera on the +y side looking back at -y.
  """
  down = cal.from_ur_pose([0, 0, 0, np.pi, 0, 0])[:3, :3]    # rotvec (pi,0,0)
  sx, sy = SIDES[side]
  # rotating about base x by -tilt moves the view direction toward -y
  axis = np.array([sy, -sx, 0.0]) * -1.0
  R_tilt = cv2.Rodrigues(axis * np.radians(tilt_deg))[0]
  return R_tilt @ down


def view_pose(center, tilt_deg, side, dist, X):
  """TCP pose so the CAMERA looks at `center` from `dist` at this tilt.

  The tool orientation is fixed by tool_rotation(); only the position is
  solved, using the hand-eye X (camera pose in the TCP frame).
  """
  T_bt = np.eye(4)
  T_bt[:3, :3] = tool_rotation(tilt_deg, side)
  R_bc = T_bt[:3, :3] @ X[:3, :3]
  cam = center - dist * R_bc[:, 2]                 # camera z points at center
  T_bt[:3, 3] = cam - T_bt[:3, :3] @ X[:3, 3]
  return T_bt


def check_view(T_bt, X, center, a):
  """Raises ValueError when a view breaks a clearance; returns the camera pose."""
  T_bc = T_bt @ X
  cam, tcp = T_bc[:3, 3], T_bt[:3, 3]
  tip = T_bt[:3, :3] @ a.tool_tip + tcp
  if np.linalg.norm(tcp[:2]) > a.max_reach:
    raise ValueError(f"TCP {np.linalg.norm(tcp[:2]):.2f} m from the base axis (> reach)")
  if np.linalg.norm(tcp[:2]) < a.min_base_dist:
    raise ValueError("TCP too close to the robot base")
  if tcp[2] < a.table_z + a.min_tcp_height:
    raise ValueError(f"TCP {1000*(tcp[2]-a.table_z):.0f} mm above the table")
  if cam[2] < a.table_z + a.min_cam_height:
    raise ValueError(f"camera {1000*(cam[2]-a.table_z):.0f} mm above the table")
  if tip[2] < a.table_z + a.min_tip_height:
    raise ValueError(f"gripper tip {1000*(tip[2]-a.table_z):.0f} mm above the table")
  if np.linalg.norm(tip - center) < a.min_tip_object:
    raise ValueError(f"gripper tip {1000*np.linalg.norm(tip-center):.0f} mm from the object")
  return T_bc


# ==============================================================================
# Finding the object, preview, prompts
# ==============================================================================
def locate(obs, T_bc, half_height):
  """Object center in the base frame from the YOLO mask and the depth."""
  ys, xs = np.nonzero(obs.mask & (obs.depth > 0))
  if len(xs) < 50:
    return None
  z = np.median(obs.depth[ys, xs]) / 1000.0
  u, v = np.median(xs), np.median(ys)
  K = obs.K
  p_cam = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
  p = T_bc[:3, :3] @ p_cam + T_bc[:3, 3]
  p[2] -= half_height        # seen surface is (about) the top; go to the center
  return p


def wait_for_operator(cam, message, gui):
  """Live YOLO preview; SPACE / Enter continues, q stops. Falls back to input()."""
  print(f"\n>>> {message}")
  if not gui or cam.simulated:
    if cam.simulated:
      print("    [sim] continuing")
      return
    if input("    Enter = continue, q = stop: ").strip().lower() == "q":
      raise KeyboardInterrupt("stopped by operator")
    return
  print("    SPACE in the preview window = continue, q = stop")
  win = "V-GAP trials: SPACE when the object is ready"
  while True:
    d, c = cam._frames()
    res = cam.yolo(c, conf=cam.settings["yolo_conf"], agnostic_nms=True, verbose=False)[0]
    img = res.plot()
    cv2.putText(img, message[:70], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    cv2.imshow(win, img)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord(" "), 13):
      cv2.destroyWindow(win)
      return
    if k in (ord("q"), 27):
      cv2.destroyWindow(win)
      raise KeyboardInterrupt("stopped by operator")


def ask_correct(obs, out, gui, sim_truth=None):
  """y/n for one representative pose. In the sim the truth answers."""
  if sim_truth is not None:
    return sim_truth
  if gui:
    img = obs.color.copy()
    for m, col in ((obs.mask, (0, 255, 0)), (out["Mproj"], (0, 0, 255))):
      cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
      cv2.drawContours(img, cs, -1, col, 2)
    cv2.putText(img, f"C={out['C']:.2f} {out['state']}  y = CAD on object, n = wrong",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    win = "Is the red outline (CAD) on the object? y / n"
    cv2.imshow(win, img)
    while True:
      k = cv2.waitKey(0) & 0xFF
      if k in (ord("y"), ord("n")):
        cv2.destroyWindow(win)
        return k == ord("y")
  while True:
    k = input("  Pose correct (CAD on the object within ~5 mm / 5 deg)? y/n: ").strip()
    if k in ("y", "n"):
      return k == "y"


# ==============================================================================
# Per-burst metrics
# ==============================================================================
def metrics_3d(scan, cad, T_c2c):
  """ICP-style fitness / RMSE at 3 mm and the share of scan points within
  3 / 5 mm of the posed CAD (same numbers show_3d_overlap prints)."""
  import open3d as o3d
  if scan is None or len(scan.points) == 0:
    return {}
  ev = o3d.pipelines.registration.evaluate_registration(
      scan, cad.pcd, 0.003, np.linalg.inv(T_c2c))
  posed = copy.deepcopy(cad.pcd).transform(T_c2c)
  d = np.asarray(scan.compute_point_cloud_distance(posed))
  return dict(fitness_3mm=round(ev.fitness, 4), rmse_mm=round(ev.inlier_rmse * 1000, 3),
              within_3mm=round(float(np.mean(d < 0.003)), 4),
              within_5mm=round(float(np.mean(d < 0.005)), 4),
              median_dist_mm=round(float(np.median(d)) * 1000, 3),
              n_scan_points=len(scan.points))


def pose_fields(T, prefix):
  rv = cv2.Rodrigues(T[:3, :3])[0].ravel()
  t = T[:3, 3] * 1000
  return {f"{prefix}_x_mm": round(t[0], 2), f"{prefix}_y_mm": round(t[1], 2),
          f"{prefix}_z_mm": round(t[2], 2), f"{prefix}_rx": round(rv[0], 5),
          f"{prefix}_ry": round(rv[1], 5), f"{prefix}_rz": round(rv[2], 5)}


def cluster_poses(Ts, cad, rot_tol=5.0, trans_tol=5.0):
  """Greedy clusters of near-identical poses (symmetry-aware). Returns labels."""
  reps, lab = [], []
  for T in Ts:
    for k, R in enumerate(reps):
      e_r, e_t = cal.pose_error(R, T, cad)
      if e_r <= rot_tol and e_t <= trans_tol:
        lab.append(k)
        break
    else:
      reps.append(T)
      lab.append(len(reps) - 1)
  return lab


METRICS = ["C", "IB", "IM", "margin", "UD", "U_live", "fitness_3mm", "rmse_mm",
           "within_3mm", "within_5mm", "median_dist_mm", "runtime_s",
           "dev_from_median_mm", "dev_from_median_deg"]


def summarize(rows, path_csv, path_txt):
  """Means / std / rates per object x angle x orientation, per angle, overall."""
  def agg(rs):
    d = dict(n=len(rs))
    ok = [r for r in rs if r.get("C") not in ("", None)]
    d["pass_rate"] = round(np.mean([r["state"] == "Valid" for r in rs]), 3) if rs else ""
    lab = [r for r in rs if r.get("label") in ("correct", "wrong")]
    d["correct_rate"] = round(np.mean([r["label"] == "correct" for r in lab]), 3) if lab else ""
    d["false_accepts"] = sum(r["state"] == "Valid" and r.get("label") == "wrong" for r in rs)
    d["false_rejects"] = sum(r["state"] != "Valid" and r.get("label") == "correct" for r in rs)
    d["fallback_rate"] = round(np.mean([bool(r.get("used_fallback")) for r in ok]), 3) if ok else ""
    d["class_swap_rate"] = round(np.mean([bool(r.get("class_swapped")) for r in ok]), 3) if ok else ""
    for m in METRICS:
      v = [float(r[m]) for r in ok if r.get(m) not in ("", None) and
           np.isfinite(float(r[m]))]
      d[f"{m}_mean"] = round(float(np.mean(v)), 4) if v else ""
      d[f"{m}_std"] = round(float(np.std(v)), 4) if v else ""
    reasons = defaultdict(int)
    for r in rs:
      if r["state"] != "Valid":
        reasons[r.get("reason") or r["state"]] += 1
    d["conflict_reasons"] = "; ".join(f"{k}: {v}" for k, v in reasons.items())
    return d

  groups = []
  keyf = [("object", "color", "angle_deg", "orientation"),
          ("object", "color", "angle_deg"), ("object", "color")]
  for keys in keyf:
    g = defaultdict(list)
    for r in rows:
      g[tuple(r[k] for k in keys)].append(r)
    for k, rs in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
      d = dict(zip(keys, k))
      d["level"] = "+".join(keys)
      d.update(agg(rs))
      groups.append(d)
  fields = ["level", "object", "color", "angle_deg", "orientation"] + \
      [k for k in groups[0] if k not in ("level", "object", "color", "angle_deg",
                                         "orientation")]
  with open(path_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(groups)
  L = [f"{'object':10s} {'color':8s} {'angle':>5s} {'ori':>3s} {'n':>3s} {'pass':>5s} "
       f"{'correct':>7s} {'FA':>3s} {'FR':>3s} {'C mean':>7s} {'C std':>6s} "
       f"{'IM':>6s} {'UD':>6s} {'rmse':>6s} {'repeat mm/deg':>14s} {'t s':>5s}"]
  for d in groups:
    L.append(f"{d['object']:10s} {d['color']:8s} {str(d.get('angle_deg', 'all')):>5s} "
             f"{str(d.get('orientation', 'all')):>3s} {d['n']:3d} "
             f"{d['pass_rate']!s:>5s} {d['correct_rate']!s:>7s} {d['false_accepts']:3d} "
             f"{d['false_rejects']:3d} {d['C_mean']!s:>7s} {d['C_std']!s:>6s} "
             f"{d['IM_mean']!s:>6s} {d['UD_mean']!s:>6s} {d['rmse_mm_mean']!s:>6s} "
             f"{str(d['dev_from_median_mm_mean'])+'/'+str(d['dev_from_median_deg_mean']):>14s} "
             f"{d['runtime_s_mean']!s:>5s}")
  txt = "\n".join(L)
  with open(path_txt, "w") as f:
    f.write("pass = gate Valid; correct = pose confirmed by you (or within 5 deg / "
            "5 mm of it); FA = Valid but wrong; FR = Conflict but correct;\n"
            "repeat = mean deviation of the 10 repeats from their median pose.\n\n")
    f.write(txt + "\n")
  return txt


# ==============================================================================
# Main
# ==============================================================================
def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--object", required=True, choices=list(vp.OBJECT_CONFIGS))
  ap.add_argument("--color", required=True)
  ap.add_argument("--robot", default="dry-run", choices=["dry-run", "ur-rtde", "sim"])
  ap.add_argument("--robot-ip", default="192.168.1.5")
  ap.add_argument("--serial", default=None, help="wrist RealSense serial")
  ap.add_argument("--hand-eye", default=None, help="T_flange_cam.npy (4x4)")
  ap.add_argument("--angles", default="0,30,45",
                  help="deg from straight down (75+ puts the wrist near the table)")
  ap.add_argument("--side", default="+y", choices=list(SIDES),
                  help="side the camera tilts toward (+y = semester_demo orbit)")
  ap.add_argument("--orientations", type=int, default=None,
                  help="object orientations per angle (default 4, hemisphere 1)")
  ap.add_argument("--reps", type=int, default=10, help="bursts per view")
  ap.add_argument("--dist", type=float, default=0.40,
                  help="camera to object center, m (D435 needs >= ~0.3)")
  ap.add_argument("--speed", type=float, default=0.10, help="moveL speed m/s")
  ap.add_argument("--tool-tip", default="0,0,0.086", help="gripper tip in TCP frame, m")
  ap.add_argument("--table-z", type=float, default=0.0, help="table height, base frame")
  ap.add_argument("--min-cam-height", type=float, default=0.08)
  ap.add_argument("--min-tip-height", type=float, default=0.03)
  ap.add_argument("--min-tcp-height", type=float, default=0.08)
  ap.add_argument("--min-tip-object", type=float, default=0.10)
  ap.add_argument("--min-base-dist", type=float, default=0.20)
  ap.add_argument("--max-reach", type=float, default=0.85)
  ap.add_argument("--target", default=None,
                  help="object center x,y,z (skips finding it; needed for --plan-only)")
  ap.add_argument("--plan-only", action="store_true")
  ap.add_argument("--no-confirm", action="store_true")
  ap.add_argument("--no-label", action="store_true", help="do not ask y/n")
  ap.add_argument("--no-inject", action="store_true")
  ap.add_argument("--no-gui", action="store_true")
  ap.add_argument("--seed", type=int, default=0, help="sim only")
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  a.tool_tip = np.asarray(cal.parse_list(a.tool_tip))
  params = vp.OBJECT_CONFIGS[a.object]
  angles = cal.parse_list(a.angles)
  n_ori = a.orientations or ORIENTATIONS[a.object]
  half_h = min(params["expected_extent_mm"]) / 2000.0
  gui = not (a.no_gui or a.robot == "sim")

  if a.hand_eye:
    X = np.load(a.hand_eye).astype(float)
  elif a.robot == "sim":
    X = np.eye(4)
    X[:3, 3] = [0.0, 0.04, -0.107]          # hand-measured mount from the research plan
  else:
    raise SystemExit("--hand-eye is needed (your T_flange_cam.npy)")

  def plan(center):
    print(f"Object center {np.round(center, 3).tolist()}, side {a.side}, "
          f"dist {a.dist} m")
    for t in angles:
      T_bt = view_pose(center, t, a.side, a.dist, X)
      try:
        check_view(T_bt, X, center, a)
        print(f"  angle {t:4.0f}: TCP {np.round(cal.to_ur_pose(T_bt), 4).tolist()}")
      except ValueError as e:
        print(f"  angle {t:4.0f}: SKIPPED ({e})")

  if a.plan_only:
    if a.target is None:
      raise SystemExit("--plan-only needs --target x,y,z")
    plan(np.asarray(cal.parse_list(a.target)))
    return

  # ------------------------------------------------------------- adapters
  if a.robot == "sim":
    import sim_stress_test as st
    for k, rel in st.MESHES.items():
      vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(st.ROOT, rel)
      vp.OBJECT_CONFIGS[k]["cad_units"] = "mm"
    cam = cal.SimCamera(a.object, a.color, a.seed, X)
    robot = cal.DryRunRobot(False, cal.from_ur_pose(OBSERVE_POSE))
  else:
    cam = cal.RealSenseCamera(a.serial)
    if a.robot == "ur-rtde":
      robot = cal.URRtdeRobot(a.robot_ip, speed=a.speed, acc=0.2,
                              confirm=not a.no_confirm)
    else:
      robot = cal.DryRunRobot(wait_for_operator=True, start_pose=cal.from_ur_pose(OBSERVE_POSE))
  quiet = cam.simulated

  cad_cache = {}
  for k in [a.object] + vp.CLASS_RIVALS.get(a.object, []):
    with contextlib.redirect_stdout(io.StringIO()):
      cad_cache[k] = vp.load_cad_model(vp.OBJECT_CONFIGS[k])
  cad = cad_cache[a.object]

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = a.out or os.path.join("data", "trials", f"{a.object}_{a.color}_{stamp}")
  os.makedirs(out_dir, exist_ok=True)
  rows_all = []
  cap_f = open(os.path.join(out_dir, "captures.csv"), "w", newline="")
  cal_f = open(os.path.join(out_dir, f"calib_{a.object}_{a.color}.csv"), "w", newline="")
  cap_w = None
  cal_w = csv.DictWriter(cal_f, fieldnames=cal.FIELDS, extrasaction="ignore")
  cal_w.writeheader()

  def move(T_bt, label):
    robot.move_tcp(T_bt, label)

  def find_object(from_pose, label):
    move(from_pose, label)
    T_bt = robot.get_tcp()
    obs = cam.capture(T_bt)
    if obs is None:
      return None
    return locate(obs, T_bt @ X, half_h)

  def sim_rotate_object():
    """Sim only: what you do by hand, a 90 deg turn about the vertical."""
    c = cam.object_center()
    T = np.eye(4)
    T[:3, :3] = vp._rot([0, 0, 1], 90.0)
    T[:3, 3] = c - T[:3, :3] @ c
    cam.T_base_obj = T @ cam.T_base_obj

  try:
    # ------------------------------------------------------------- find it
    if a.target:
      center = np.asarray(cal.parse_list(a.target))
    else:
      print("[*] Moving to OBSERVE_POSE to find the object")
      center = find_object(cal.from_ur_pose(OBSERVE_POSE), "observe pose")
      if center is None:
        raise SystemExit("No object found from OBSERVE_POSE; pass --target x,y,z")
    print(f"[+] Object at {np.round(center, 3).tolist()} (base frame)")
    plan(center)

    ori_now = 0
    for ai, ang in enumerate(angles):
      # re-find it from straight above (you may have nudged it while turning)
      above = view_pose(center, 0.0, a.side, a.dist, X)
      c2 = find_object(above, f"above object (angle {ang:g} next)")
      if c2 is not None and np.linalg.norm(c2[:2] - center[:2]) < 0.10:
        center = c2
      T_bt = view_pose(center, ang, a.side, a.dist, X)
      try:
        check_view(T_bt, X, center, a)
      except ValueError as e:
        print(f"[-] angle {ang:g} skipped: {e}")
        continue
      move(T_bt, f"angle {ang:g} deg")

      for oi in range(n_ori):
        if not (ai == 0 and oi == 0):
          turns = (oi - ori_now) % n_ori if n_ori > 1 else 0
          msg = (f"Angle {ang:g}: put the {a.object} in orientation {oi + 1}/{n_ori} "
                 f"(turn it {90 * turns} deg about its vertical axis, same spot)") \
              if turns else f"Angle {ang:g}: check the {a.object} is in place"
          if cam.simulated:
            for _ in range(turns):
              sim_rotate_object()
          wait_for_operator(cam, msg, gui)
          ori_now = oi
        group = []
        for rep in range(a.reps):
          T_bt = robot.get_tcp()
          obs = cam.capture(T_bt)
          base = dict(object=a.object, color=a.color, angle_deg=ang,
                      orientation=oi + 1, rep=rep + 1,
                      timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                      tcp_pose=json.dumps([round(v, 6) for v in cal.to_ur_pose(T_bt)]))
          if obs is None:
            r = dict(base, state="no detection", reason="no detection", C="")
            group.append(dict(row=r))
            print(f"  [{ang:g} deg / ori {oi+1} / {rep+1}] no detection")
            continue
          out = cal.run_pipeline(obs, cad_cache, quiet)
          fin = out.get("final_object", obs.object_key)
          r = dict(base, yolo_object=obs.object_key, yolo_conf=round(obs.confidence, 3),
                   final_object=fin, class_swapped=out.get("class_swapped", False),
                   class_margin=out.get("class_margin", ""),
                   class_scores=json.dumps(out.get("class_scores", {})),
                   state=out["state"], reason=out.get("reason", ""),
                   C=round(out.get("C", 0.0), 4), IB=round(out.get("IB", 0.0), 4),
                   IM=round(out.get("IM", 0.0), 4),
                   margin=round(out.get("margin", float("nan")), 4),
                   UD=round(out.get("UD", float("nan")), 4),
                   U_live=round(out.get("U_live", float("nan")), 4),
                   used_fallback=out.get("used_fallback", False),
                   chosen=out.get("chosen", ""), all_scores=json.dumps(out.get("all_scores", [])),
                   axis_ambiguity_deg=out.get("axis_ambiguity_deg"),
                   runtime_s=round(out.get("runtime_s", float("nan")), 2),
                   mask_px=int(obs.mask.sum()))
          if "T_cad_to_cam" in out:
            Tcc = out["T_cad_to_cam"]
            r.update(metrics_3d(out.get("scan"), cad_cache[fin], Tcc))
            r.update(pose_fields(Tcc, "cam"))
            r.update(pose_fields(T_bt @ X @ Tcc, "base"))
          if not cam.simulated and "T_cad_to_cam" in out:
            d = os.path.join(out_dir, "trials")
            obs.meta["timestamp"] = f"a{ang:g}_o{oi+1}_r{rep+1}_{obs.meta['timestamp']}"
            with contextlib.redirect_stdout(io.StringIO()):
              vp.save_trial(obs, out, root=d)
            r["trial_dir"] = os.path.join(d, f"{obs.object_key}_{obs.meta['timestamp']}")
          if cam.simulated and "T_cad_to_cam" in out:
            tr, tt = cal.pose_error(cam.true_pose_cam(T_bt), out["T_cad_to_cam"], cad)
            r.update(true_rot_err_deg=round(tr, 2), true_trans_err_mm=round(tt, 2))
          group.append(dict(row=r, obs=obs, out=out))
          print(f"  [{ang:g} deg / ori {oi+1} / {rep+1}] {fin}: {r['state']} "
                f"C={r['C']:.3f} {r['reason']}")

        # ---- labels: one question per distinct pose, then by distance
        posed = [g for g in group if "out" in g and "T_cad_to_cam" in g["out"]
                 and g["out"].get("final_object") == a.object]
        T_ref = None
        if posed:
          Ts = [g["out"]["T_cad_to_cam"] for g in posed]
          med_t = np.median([T[:3, 3] for T in Ts], axis=0)
          i_med = int(np.argmin([np.linalg.norm(T[:3, 3] - med_t) for T in Ts]))
          for g, T in zip(posed, Ts):
            e_r, e_t = cal.pose_error(Ts[i_med], T, cad)
            g["row"].update(dev_from_median_mm=round(e_t, 2),
                            dev_from_median_deg=round(e_r, 2))
          if not a.no_label:
            lab = cluster_poses(Ts, cad)
            for k in sorted(set(lab), key=lambda k: -lab.count(k)):
              g = posed[lab.index(k)]
              truth = None
              if cam.simulated:
                truth = float(g["row"]["true_rot_err_deg"]) <= 5 and \
                    float(g["row"]["true_trans_err_mm"]) <= 5
              print(f"  pose group {k + 1}: {lab.count(k)} of {len(Ts)} repeats, "
                    f"C={g['row']['C']:.3f} {g['row']['state']}")
              if ask_correct(g["obs"], g["out"], gui, truth):
                T_ref = g["out"]["T_cad_to_cam"]
                break
        for g in group:
          r = g["row"]
          if "out" not in g or "T_cad_to_cam" not in g["out"]:
            r.update(label="wrong", label_source="no pose")
          elif g["out"].get("final_object") != a.object:
            r.update(label="wrong", label_source="wrong class")
          elif T_ref is not None:
            e_r, e_t = cal.pose_error(T_ref, g["out"]["T_cad_to_cam"], cad)
            r.update(rot_err_deg=round(e_r, 2), trans_err_mm=round(e_t, 2),
                     label=cal.label_of(e_r, e_t), label_source="confirmed pose")
          elif not a.no_label:
            r.update(label="wrong", label_source="no pose confirmed")
          else:
            r.update(label="", label_source="")
          rows_all.append(r)
          if cap_w is None:
            keys = list(r.keys()) + [k for k in (
                "fitness_3mm", "rmse_mm", "within_3mm", "within_5mm", "median_dist_mm",
                "n_scan_points", "dev_from_median_mm", "dev_from_median_deg",
                "rot_err_deg", "trans_err_mm", "true_rot_err_deg", "true_trans_err_mm",
                "trial_dir", "label", "label_source") if k not in r]
            for p in ("cam", "base"):
              keys += [f"{p}_{s}" for s in ("x_mm", "y_mm", "z_mm", "rx", "ry", "rz")
                       if f"{p}_{s}" not in keys]
            cap_w = csv.DictWriter(cap_f, fieldnames=keys, extrasaction="ignore")
            cap_w.writeheader()
          cap_w.writerow(r)
          if r.get("C") not in ("", None) and r.get("label") in ("correct", "wrong",
                                                                 "borderline"):
            cal_w.writerow(dict(r, view=f"a{ang:g}_o{oi+1}", tilt_deg=ang,
                                azimuth_deg=90 * oi, sample="real",
                                caught_by_other_check=cal.caught_by_other_check(
                                    g["out"], params)))
        # ---- injected errors around the confirmed pose (once per view)
        if T_ref is not None and not a.no_inject:
          g = next(g for g in posed if cal.pose_error(
              T_ref, g["out"]["T_cad_to_cam"], cad)[1] <= 5)
          with contextlib.redirect_stdout(io.StringIO()):
            inj = cal.injected_rows(g["obs"], g["out"].get("scan"), cad, params, T_ref)
          for ir in inj:
            cal_w.writerow(dict(object=a.object, color=a.color, view=f"a{ang:g}_o{oi+1}",
                                tilt_deg=ang, azimuth_deg=90 * oi,
                                yolo_object=a.object, final_object=a.object, **ir))
        cap_f.flush()
        cal_f.flush()
        n_ok = sum(g["row"].get("state") == "Valid" for g in group)
        print(f"  => angle {ang:g}, orientation {oi+1}: {n_ok}/{len(group)} passed the gate")
  except KeyboardInterrupt:
    print("\n[!] Stopped; everything so far is saved.")
  finally:
    cap_f.close()
    cal_f.close()
    with contextlib.suppress(Exception):
      if not cam.simulated:
        robot.move_tcp(cal.from_ur_pose(OBSERVE_POSE), "back to OBSERVE_POSE")
    robot.close()
    cam.close()
  if rows_all:
    txt = summarize(rows_all, os.path.join(out_dir, "summary_by_view.csv"),
                    os.path.join(out_dir, "summary.txt"))
    print("\n" + txt)
  print(f"\n[+] Saved to {out_dir}\n    After all objects: python recommend_tau_c.py "
        f"data/trials/*/calib_*.csv --sweep-weights")


if __name__ == "__main__":
  main()
