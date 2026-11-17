"""V-GAP gate evaluation on the UR5e: does the IoU gate do its job?

Same flow as vgap_trials_ur5e.py (find the object, tilt in one plane, you
turn the object, 10 bursts per view), but instead of picking tau_c it TESTS
a fixed gate (the tau_c in OBJECT_CONFIGS, or --tau-c):

  0. At each angle you do all orientations flat on the table, then all of
     them again propped up on a screw (--placements; the propped pass is
     where the hemisphere's axis check and the table yaw-sweep fallback get
     tested, since both assume the object sits flat).
  1. Every burst runs the full pipeline and the gate decides Valid / Conflict.
  2. You grade every distinct pose the pipeline produced at that view
     (y = CAD on the object, b = a bit off, n = wrong). Repeats that land on
     the same pose (within 5 deg / 5 mm) share the grade, so you answer once
     per distinct pose, not once per burst.
  3. Each burst gets an outcome:
       TA  accepted and correct            (gate did its job)
       FA  accepted but wrong              (gate FAILED: bad pose let through)
       AO  accepted but a bit off (5-10 deg / 5-10 mm)
       TR  rejected and wrong              (gate did its job)
       FR  rejected but correct            (gate too strict: lost a good pose)
       RO  rejected and a bit off
  4. Accepted poses are also scored for accuracy: distance from the pose you
     confirmed, 3D overlap with the scan (share of scan points within 3 mm,
     RMSE), and repeatability over the 10 bursts.

Tests in the report (gate_report.txt):
  A  Gate confusion matrix per object, color, angle, orientation, condition,
     with 95% confidence intervals (0 false accepts in 160 still means "up to
     about 2.3%", which is what you can claim in the paper).
  B  Which check rejected what (C below tau_c, ambiguous orientation, class,
     hemisphere axis) and how many of those rejections were good poses.
  C  Accuracy of the accepted poses.
  D  Gate on vs off: how many wrong poses would reach the gripper without it.
  E  Adversarial test: wrong poses made on purpose around your confirmed pose
     (yaw 15/30/90/180, flip, x/y/z shifts, tilt), raw and after ICP. The
     pipeline rarely fails by itself, so this is where the gate is tested
     against hard wrong poses.
  F  Proposed 3D check (share of scan points within 3 mm of the posed CAD):
     what it would add to C, swept over thresholds, on real and adversarial
     poses. C alone cannot see depth and tilt errors.
  G  Stress conditions (--conditions): partly covered, a second object next
     to it, near the image edge, partly out of view, dim light. These are the
     cases where the gate should reject or the pipeline is most likely to
     fail.
  H  Recovery (--retry): on a Conflict where YOLO found the object, first
     one fresh burst from the same view (--same-view-retries, no arm move;
     0 turns it off). If that is still a Conflict, or YOLO found nothing,
     the camera tilts --retry-step deg (or to the best angle, --retry-best) and
     tries once more. With --retry-orbit, if that is still not Valid it
     circles the object counter-clockwise (seen from above) at --orbit-tilt,
     stopping every --orbit-step deg (--orbit-stops stops, default one 90 deg
     turn; a second turn is out of the arm's reach) to try again; it moves
     along the arc in small steps. If that is still not Valid and the first
     view was not straight down, it tries once from straight above (an
     occluder may block both side views); --no-overhead turns that off. Reports how
     often each step gives a correct Valid pose, per condition (use it with
     --conditions occluded: block the first view on purpose).
  I  Physical accuracy (--point-check): the gripper tip goes to hover above
     the top of the object where the pipeline says it is, and you measure
     how far the tip is from the real center (mark the top center with a
     dot first). This includes the hand-eye error, so it is the number that
     matters for a grasp.
  J  YOLO class accuracy and class-check swaps, runtime per burst.
  K  V-GAP vs plain RANSAC+ICP on the same captures: the RANSAC+ICP pose
     before the hypothesis check / fallback / class check is logged for
     every burst (you grade it when it differs from V-GAP's pose). Shows how
     often each of those steps rescued a wrong pose (or broke a good one) and
     how many wrong poses each method would send to the gripper.

Outputs in --out (default data/gate_eval/<object>_<color>_<time>/):
  gate_eval.csv, injected.csv, recovery.csv, point_check.csv, gate_report.txt,
  gate_summary.csv, trials/ (images + result.json of every burst).
Combine runs (all objects / colors) into one report:
  python vgap_gate_eval_ur5e.py --report data/gate_eval/*/

Motion safety is the same as vgap_trials_ur5e.py: slow moveL, Enter before
each move unless --no-confirm, views that break a clearance are skipped.
--point-check brings the gripper tip down to --hover above the object top;
watch the first one with a hand on the stop. UNTESTED ON HARDWARE.

Example:
  python vgap_gate_eval_ur5e.py --object gearbox --color white --robot ur-rtde \
      --serial 337122075128 --hand-eye matrix_config/T_flange_cam.npy \
      --cad cad/gearbox/fixed/origin_centered_local_global_153mm.pcd \
      --conditions normal,occluded,distractor --retry --point-check
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vgap_pipeline as vp  # noqa: E402
import calibrate_tau_c_ur5e as cal  # noqa: E402
import vgap_trials_ur5e as tr  # noqa: E402  (views, prompts, 3D metrics)

CONDITIONS = {
    "normal": "normal setup: nothing covering it, no other objects",
    "occluded": "cover about a quarter of the object (hand, card or paper), keep it still",
    "distractor": "put a second object next to it (about 2 cm away)",
    "offcenter": "slide it about 8 cm toward the edge of the image, fully in view",
    "partial": "slide it so about a third of it is OUT of the image",
    "dim": "dim the light on it (room lights off, or shade it)",
}
PLACEMENTS = {
    "flat": "flat on the table",
    "propped": "propped up on a screw (one edge resting on it, so it sits tilted)",
}
# Gocator reference columns (--gocator); errors are V-GAP's pose vs the
# Gocator pose, both in the base frame, symmetry-aware
GT_KEYS = ["gt_status", "gt_rot_err_deg", "gt_trans_err_mm", "gt_grade", "gt_outcome",
           "gt_fitness_1mm", "gt_rmse_mm", "gt_margin", "gt_repeat_mm", "gt_repeat_deg",
           "gt_n_points", "gt_rot_valid", "baseline_gt_rot_err_deg",
           "baseline_gt_trans_err_mm", "baseline_gt_grade"]

# per-burst detection filter log (vp.select_detection)
DET_FILTER_EMPTY = dict(n_detections="", n_dropped_gripper="", n_dropped_far="",
                        n_dropped_unknown="", target_dist_mm="", dropped="")
OUTCOMES = ["TA", "FA", "AO", "TR", "FR", "RO", "no detection"]
THR_3D = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ==============================================================================
# Plain RANSAC+ICP baseline: the pose BEFORE the gate's hypothesis check,
# table fallback and class check (the first pose iou_check() is handed)
# ==============================================================================
_BASELINE = {}
_orig_iou_check = vp.iou_check
_orig_estimate_pose = vp.estimate_pose


def _iou_check_spy(obs, scan, cad, T_scan_to_cad, params):
  _BASELINE.setdefault(id(cad), T_scan_to_cad.copy())
  return _orig_iou_check(obs, scan, cad, T_scan_to_cad, params)


def _estimate_pose_spy(obs, cad, params):
  _BASELINE.pop(id(cad), None)
  return _orig_estimate_pose(obs, cad, params)


vp.iou_check = _iou_check_spy
vp.estimate_pose = _estimate_pose_spy


class StaticRobot:
  """--static: never connects to the arm and never moves. get_tcp() is the
  fixed pose you give (--static-tcp) or identity."""
  moves_hardware = False

  def __init__(self, T):
    self.T = T

  def get_tcp(self):
    return self.T.copy()

  def move_tcp(self, T_base_tool, label=""):
    raise RuntimeError(f"--static: tried to move the arm ({label})")

  def close(self):
    pass


def baseline_pose(obs, cad_cache):
  """Plain RANSAC+ICP pose for YOLO's class (cam frame), or None."""
  cad = cad_cache.get(obs.object_key)
  T = _BASELINE.get(id(cad)) if cad is not None else None
  return None if T is None else np.linalg.inv(T)


# ==============================================================================
# Grading and outcomes
# ==============================================================================
def ask_grade(obs, out, gui, sim_grade=None, note=""):
  """y / b / n for one pose -> correct / borderline / wrong."""
  if sim_grade is not None:
    return sim_grade
  keys = {"y": "correct", "b": "borderline", "n": "wrong"}
  if gui:
    img = obs.color.copy()
    for m, col in ((obs.mask, (0, 255, 0)), (out.get("Mproj"), (0, 0, 255))):
      if m is None:
        continue
      cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
      cv2.drawContours(img, cs, -1, col, 2)
    cv2.putText(img, f"{out.get('final_object', '')} C={out.get('C', 0):.2f} "
                f"{out.get('state', '')} {note}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "y = on the object, b = a bit off (5-10 mm / deg), n = wrong",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    win = "Grade the red outline (CAD): y / b / n"
    cv2.imshow(win, img)
    while True:
      k = chr(cv2.waitKey(0) & 0xFF)
      if k in keys:
        cv2.destroyWindow(win)
        return keys[k]
  while True:
    k = input("  CAD on the object? y = yes, b = a bit off, n = wrong: ").strip()
    if k in keys:
      return keys[k]


def outcome(state, grade):
  if state == "no detection":
    return "no detection"
  acc = state == "Valid"
  return {("correct", True): "TA", ("wrong", True): "FA", ("borderline", True): "AO",
          ("correct", False): "FR", ("wrong", False): "TR",
          ("borderline", False): "RO"}[(grade, acc)]


# Recovery step 1 target per object (--retry-best): the tilt with the most
# correct accepted poses in the 10/22 gate runs (TA share at 0 / 30 / 45 deg):
# gearbox 89 / 100 / 96%, L 87 / 96 / 88%, T 94 / 98 / 87%, hemisphere
# 78 / 68 / 76%. The hemisphere orbit uses --orbit-tilt, since an orbit at
# 0 deg is the same view.
BEST_ANGLES_DEFAULT = "gearbox:30,l_block:30,t_block:30,hemisphere:0"


# Azimuth (deg, counter-clockwise from base +x, seen from above) of each
# --side; view_pose_az() at these azimuths gives exactly tr.view_pose().
SIDE_AZ = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}


def view_pose_az(center, tilt_deg, az_deg, dist, X):
  """tr.view_pose() for any azimuth around the object: straight down tilted
  toward the side at az_deg. The tool does not spin about the vertical, so
  the wrist only tilts as the camera goes around."""
  down = cal.from_ur_pose([0, 0, 0, np.pi, 0, 0])[:3, :3]
  sx, sy = np.cos(np.radians(az_deg)), np.sin(np.radians(az_deg))
  axis = np.array([sy, -sx, 0.0]) * -1.0
  T_bt = np.eye(4)
  T_bt[:3, :3] = cv2.Rodrigues(axis * np.radians(tilt_deg))[0] @ down
  R_bc = T_bt[:3, :3] @ X[:3, :3]
  cam = center - dist * R_bc[:, 2]
  T_bt[:3, 3] = cam - T_bt[:3, :3] @ X[:3, 3]
  return T_bt


def wilson(k, n, z=1.96):
  if n == 0:
    return None, None
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
  if n == 0:
    return "   -  "
  lo, hi = wilson(k, n)
  return f"{100 * k / n:5.1f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


# ==============================================================================
# Adversarial poses (test E/F)
# ==============================================================================
def injected_eval(obs, scan, cad, params, T_ref):
  """Wrong poses around the confirmed one, raw and after ICP, with C, the
  gate decision (C only; margin needs the full hypothesis search) and the 3D
  overlap. ICP results that did not move are dropped."""
  rows = []
  ax = {"long": 0, "mid": 1, "height": 2}
  for name, kind, what, amt, group in cal.INJECTIONS:
    if kind == "rot":
      if cad.axial_axis is not None and what == "height":
        continue
      T = T_ref @ vp._rot_about(cad.center, cad.axes[:, ax[what]], amt)
    else:
      T = T_ref.copy()
      T[:3, 3] += amt * np.asarray(what, float)
    variants = [("raw", T)]
    if params["icp_thresholds"] and scan is not None and len(scan.points) >= 50:
      with contextlib.redirect_stdout(io.StringIO()):
        T_s2c, _, _ = vp.icp_refine(scan, cad, np.linalg.inv(T), params)
      Ti = np.linalg.inv(T_s2c)
      if cal.pose_error(T, Ti, cad)[1] > 0.1 or \
          vp.symmetric_rotation_error_deg(T[:3, :3], Ti[:3, :3], cad) > 0.05:
        variants.append(("icp", Ti))
    for tag, Tv in variants:
      IB, IM, C = cal.score_pose(obs, cad, Tv, params)
      e_r, e_t = cal.pose_error(T_ref, Tv, cad)
      r = dict(perturbation=name, inject_kind=tag, inject_group=group,
               C=round(C, 4), IB=round(IB, 4), IM=round(IM, 4),
               rot_err_deg=round(e_r, 2), trans_err_mm=round(e_t, 2),
               grade=cal.label_of(e_r, e_t),
               gate=("Valid" if C >= params["tau_c"] else "Conflict"))
      r.update(tr.metrics_3d(scan, cad, Tv))
      rows.append(r)
  return rows


# ==============================================================================
# Report (works on the CSVs, so runs can be combined later)
# ==============================================================================
def _f(v):
  try:
    x = float(v)
    return x if np.isfinite(x) else None
  except (TypeError, ValueError):
    return None


def _read(paths):
  rows = []
  for p in paths:
    if os.path.exists(p):
      with open(p) as f:
        rows += list(csv.DictReader(f))
  return rows


def confusion(rs):
  c = Counter(r["outcome"] for r in rs)
  d = {k: c.get(k, 0) for k in OUTCOMES}
  d["n"] = len(rs)
  d["accepted"] = d["TA"] + d["FA"] + d["AO"]
  d["wrong"] = d["FA"] + d["TR"]
  d["good"] = d["TA"] + d["FR"]
  return d


def stats(vals):
  v = [x for x in (_f(x) for x in vals) if x is not None]
  if not v:
    return "-"
  return f"{np.mean(v):.2f} +/- {np.std(v):.2f} (max {np.max(v):.2f}, n {len(v)})"


def report(dirs, out_txt, out_csv):
  R = _read([os.path.join(d, "gate_eval.csv") for d in dirs])
  INJ = _read([os.path.join(d, "injected.csv") for d in dirs])
  REC = _read([os.path.join(d, "recovery.csv") for d in dirs])
  PC = _read([os.path.join(d, "point_check.csv") for d in dirs])
  if not R:
    raise SystemExit("no gate_eval.csv rows found")
  L = []
  P = L.append
  P("V-GAP gate evaluation")
  P(f"runs: {len(dirs)}   bursts: {len(R)}   objects: "
    f"{sorted(set(r['object'] for r in R))}")
  taus = sorted(set((r["object"], r["tau_c"]) for r in R))
  P("tau_c used: " + ", ".join(f"{o} {t}" for o, t in taus))
  P("TA accepted+correct, FA accepted+WRONG, AO accepted+a bit off, TR rejected+wrong,")
  P("FR rejected+correct, RO rejected+a bit off. Rates show 95% Wilson intervals.")
  P("FA rate = FA / all wrong poses (how often a wrong pose gets through).")
  P("Precision = TA / accepted (how often an accepted pose is right).")
  P("FR rate = FR / all correct poses (good poses thrown away).\n")

  # ---- A: confusion tables
  summary_rows = []
  levels = [("object",), ("object", "color"), ("object", "placement"),
            ("object", "placement", "angle_deg"), ("object", "angle_deg"),
            ("object", "orientation"), ("object", "condition"),
            ("object", "color", "placement", "angle_deg", "condition")]
  P("A. Gate confusion")
  for keys in levels:
    g = defaultdict(list)
    for r in R:
      g[tuple(r[k] for k in keys)].append(r)
    P(f"\n  by {' x '.join(keys)}")
    P(f"  {'group':34s} {'n':>4s} {'TA':>4s} {'FA':>3s} {'AO':>3s} {'TR':>4s} "
      f"{'FR':>3s} {'RO':>3s} {'nd':>3s}  {'FA rate':17s} {'precision':17s} {'FR rate':17s}")
    for k, rs in sorted(g.items(), key=lambda kv: [str(x) for x in kv[0]]):
      d = confusion(rs)
      P(f"  {' / '.join(k):34s} {d['n']:4d} {d['TA']:4d} {d['FA']:3d} {d['AO']:3d} "
        f"{d['TR']:4d} {d['FR']:3d} {d['RO']:3d} {d['no detection']:3d}  "
        f"{pct(d['FA'], d['wrong']):17s} {pct(d['TA'], d['accepted']):17s} "
        f"{pct(d['FR'], d['good']):17s}")
      summary_rows.append(dict(level="+".join(keys), group=" / ".join(k), **d))

  # ---- B: which check rejected
  P("\nB. Rejections by reason (what the rejected poses really were)")
  g = defaultdict(Counter)
  for r in R:
    if r["state"] not in ("Valid", "no detection"):
      g[(r["object"], r["reason"] or r["state"])][r["grade"]] += 1
  for (o, why), c in sorted(g.items()):
    P(f"  {o:10s} {why:26s} total {sum(c.values()):4d}: wrong {c['wrong']}, "
      f"correct {c['correct']} (FR), a bit off {c['borderline']}")

  # ---- C: accuracy of accepted poses
  P("\nC. Accuracy of ACCEPTED poses (mean +/- std)")
  for o in sorted(set(r["object"] for r in R)):
    acc = [r for r in R if r["object"] == o and r["state"] == "Valid"]
    if not acc:
      continue
    gr = Counter(r["grade"] for r in acc)
    P(f"  {o}: {len(acc)} accepted -> correct {gr['correct']}, a bit off "
      f"{gr['borderline']}, wrong {gr['wrong']}")
    P(f"    offset from your confirmed pose, mm : {stats(r['dev_ref_mm'] for r in acc)}")
    P(f"    offset from your confirmed pose, deg: {stats(r['dev_ref_deg'] for r in acc)}")
    P(f"    repeat spread (10 bursts), mm       : {stats(r['dev_from_median_mm'] for r in acc)}")
    P(f"    repeat spread (10 bursts), deg      : {stats(r['dev_from_median_deg'] for r in acc)}")
    P(f"    scan points within 3 mm of CAD      : {stats(r.get('within_3mm') for r in acc)}")
    P(f"    ICP RMSE at 3 mm, mm                : {stats(r.get('rmse_mm') for r in acc)}")
    P(f"    C                                   : {stats(r['C'] for r in acc)}")
    for ang in sorted(set(r["angle_deg"] for r in acc), key=float):
      a2 = [r for r in acc if r["angle_deg"] == ang]
      P(f"    angle {ang:>3s}: offset mm {stats(r['dev_ref_mm'] for r in a2)}")

  # ---- D: gate on vs off
  P("\nD. Gate on vs off (bursts with a pose)")
  for o in sorted(set(r["object"] for r in R)):
    rs = [r for r in R if r["object"] == o and r["state"] != "no detection"
          and r.get("has_pose") == "True"]
    if not rs:
      continue
    w_off = sum(r["grade"] == "wrong" for r in rs)
    acc = [r for r in rs if r["state"] == "Valid"]
    w_on = sum(r["grade"] == "wrong" for r in acc)
    P(f"  {o:10s} gate OFF: {w_off}/{len(rs)} wrong poses sent on "
      f"({pct(w_off, len(rs)).strip()}); gate ON: {w_on}/{len(acc)} "
      f"({pct(w_on, len(acc)).strip()}); good poses lost to the gate: "
      f"{sum(r['grade'] == 'correct' and r['state'] != 'Valid' for r in rs)}")

  # ---- E: adversarial
  if INJ:
    P("\nE. Adversarial wrong poses: share REJECTED by C (raw = no ICP, icp = after ICP)")
    P("   only poses that are really wrong (> 10 deg or > 10 mm) count; ICP often "
      "fixes small errors, those are listed as 'fixed by ICP'")
    g = defaultdict(list)
    for r in INJ:
      g[(r["object"], r.get("placement", "flat") + " " + r["perturbation"],
       r["inject_kind"])].append(r)
    for (o, p, k), rs in sorted(g.items()):
      wr = [r for r in rs if r["grade"] == "wrong"]
      fixed = sum(r["grade"] == "correct" for r in rs)
      caught = sum(r["gate"] != "Valid" for r in wr)
      P(f"  {o:10s} {p:22s} {k:4s} wrong {len(wr):3d}  rejected by C "
        f"{pct(caught, len(wr)):17s}  fixed by ICP {fixed}")

  # ---- F: 3D check
  P("\nF. Proposed 3D check: also require 'scan points within 3 mm' >= thr")
  for o in sorted(set(r["object"] for r in R)):
    nat = [r for r in R if r["object"] == o and _f(r.get("within_3mm")) is not None]
    inj = [r for r in INJ if r["object"] == o and r["grade"] == "wrong"
           and _f(r.get("within_3mm")) is not None]
    if not nat:
      continue
    good = [r for r in nat if r["grade"] == "correct"]
    P(f"  {o}: within_3mm on correct poses {stats(r['within_3mm'] for r in good)}")
    P(f"    {'thr':>4s}  {'real FA':>7s} {'real FR':>7s} {'adv wrong passed':>17s} "
      f"{'adv depth/tilt passed':>22s}")
    for thr in [None] + THR_3D:
      def ok(r):
        c = r.get("state", r.get("gate")) == "Valid"
        return c and (thr is None or _f(r["within_3mm"]) >= thr)
      fa = sum(ok(r) and r["grade"] == "wrong" for r in nat)
      fr = sum(not ok(r) and r["grade"] == "correct" for r in nat)
      adv = sum(ok(r) for r in inj)
      oop = [r for r in inj if r["inject_group"] == "out_of_plane"]
      P(f"    {('C only' if thr is None else f'{thr:.1f}'):>6s}  {fa:7d} {fr:7d} "
        f"{adv:7d}/{len(inj):<9d} {sum(ok(r) for r in oop):9d}/{len(oop):<12d}")

  # ---- G is in A (by condition)
  # ---- H: recovery
  if REC:
    P("\nH. Recovery after a Conflict (0 = same view again, 1 = tilt / best angle, "
      "2 = 90 deg counter-clockwise, 3 = straight above)")
    g = defaultdict(list)
    for r in REC:
      g[r["object"]].append(r)
    for o, rs in sorted(g.items()):
      chains = defaultdict(list)
      for r in rs:
        chains[r.get("conflict_id") or id(r)].append(r)
      n = len(chains)
      P(f"  {o}: {n} Conflicts retried")
      for stage in ("same", "tilt", "orbit", "overhead"):
        st = [r for r in rs if r.get("stage", "tilt") == stage]
        if st:
          ok = sum(r["new_outcome"] == "TA" for r in st)
          P(f"    {stage:6s} attempts {len(st):3d}: correct Valid {pct(ok, len(st)).strip()}, "
            f"wrong Valid {sum(r['new_outcome'] == 'FA' for r in st)}")
      fin = [c[-1] for c in chains.values()]
      ok = sum(r["new_outcome"] == "TA" for r in fin)
      P(f"    after all recovery steps: correct Valid {pct(ok, n).strip()}, wrong Valid "
        f"{sum(r['new_outcome'] == 'FA' for r in fin)}, still no Valid pose "
        f"{sum(r['new_state'] != 'Valid' for r in fin)}")
      by_cond = defaultdict(list)
      for c in chains.values():
        by_cond[c[0].get("condition", "normal")].append(c[-1])
      for cond, f in sorted(by_cond.items()):
        P(f"      {cond:10s} {sum(r['new_outcome'] == 'TA' for r in f)}/{len(f)} recovered")

  # ---- I: point check
  if PC:
    P("\nI. Physical accuracy: gripper tip above the predicted top center")
    g = defaultdict(list)
    for r in PC:
      g[r["object"]].append(r)
    for o, rs in sorted(g.items()):
      P(f"  {o:10s} n {len(rs)}  horizontal offset mm {stats(r['xy_err_mm'] for r in rs)}"
        f"  height gap error mm {stats(r['z_err_mm'] for r in rs)}")

  # ---- J: class + runtime
  P("\nJ. Class and runtime")
  for o in sorted(set(r["object"] for r in R)):
    rs = [r for r in R if r["object"] == o and r["state"] != "no detection"]
    if not rs:
      continue
    yo = sum(r["yolo_object"] == o for r in rs)
    fo = sum(r["final_object"] == o for r in rs)
    sw = [r for r in rs if r.get("class_swapped") == "True"]
    P(f"  {o:10s} YOLO class right {pct(yo, len(rs)).strip()}, after class check "
      f"{pct(fo, len(rs)).strip()}, swaps {len(sw)} (right "
      f"{sum(r['final_object'] == o for r in sw)}), no detection "
      f"{sum(r['state'] == 'no detection' for r in R if r['object'] == o)}")
    P(f"             runtime s {stats(r['runtime_s'] for r in rs)}")

  # ---- K: V-GAP vs plain RANSAC+ICP
  P("\nK. V-GAP vs plain RANSAC+ICP (same captures)")
  P("   plain = RANSAC+ICP pose for YOLO's class, sent to the gripper with no gate;")
  P("   V-GAP = after hypothesis check, table fallback, class check, then the gate.")
  for o in sorted(set(r["object"] for r in R)):
    rs = [r for r in R if r["object"] == o and r.get("baseline_grade")]
    if not rs:
      continue
    n = len(rs)
    b_ok = sum(r["baseline_grade"] == "correct" for r in rs)
    b_bad = sum(r["baseline_grade"] == "wrong" for r in rs)
    f_ok = sum(r["grade"] == "correct" for r in rs)
    sent = [r for r in rs if r["state"] == "Valid"]
    s_bad = sum(r["grade"] == "wrong" for r in sent)
    P(f"  {o}: {n} captures")
    P(f"    plain RANSAC+ICP pose correct   {pct(b_ok, n)}   wrong poses sent to the gripper: {b_bad}")
    P(f"    V-GAP final pose correct        {pct(f_ok, n)}")
    P(f"    V-GAP after the gate: sent {len(sent)}, correct "
      f"{sum(r['grade'] == 'correct' for r in sent)}, wrong poses sent: {s_bad}")
    P(f"    wrong poses reaching the gripper: plain {b_bad}/{n} vs V-GAP {s_bad}/{n}")
    P(f"    {'used':>28s} {'n':>4s} {'rescued':>8s} {'harmed':>7s} {'same':>5s}")
    for mech, key in (("90/180 deg hypothesis", "hypothesis"),
                      ("table yaw-sweep fallback", "fallback"),
                      ("L/T class check", "class check"), ("other change", "other")):
      m = [r for r in rs if r.get("changed_by") == key]
      if m:
        c = Counter(r["vgap_effect"] for r in m)
        P(f"    {mech:>28s} {len(m):4d} {c['rescued']:8d} {c['harmed']:7d} {c['same']:5d}")
    caught = sum(r["baseline_grade"] == "wrong" and r["state"] != "Valid"
                 and r["grade"] != "correct" for r in rs)
    P(f"    plain-wrong poses that V-GAP could not fix but the gate rejected: {caught}")

  # ---- L: Gocator reference (--gocator)
  def _gt_bad(r):   # also re-checks rows written before the in-run guard existed
    st = r.get("gt_status") or ""
    if st and st != "ok":
      return st
    f, rd, rm = _f(r.get("gt_fitness_1mm")), _f(r.get("gt_repeat_deg")), _f(r.get("gt_repeat_mm"))
    if f is not None and f < 0.95:
      return f"invalid: fitness {f}"
    if (rd or 0) > 2 or (rm or 0) > 2:
      return f"invalid: repeat {rd} deg / {rm} mm"
    return ""
  G = [r for r in R if r.get("gt_grade") and not _gt_bad(r)]
  bad_v = sorted({(r["object"], r["view"], _gt_bad(r)) for r in R
                  if r.get("gt_status") and _gt_bad(r)})
  if G or bad_v:
    P("\nL. Gocator 3210 reference (objective grading: correct <= 5 deg & 5 mm, "
      "wrong > 10; propped hemisphere = center only)")
    for o, v, st in bad_v:
      P(f"  {o} {v}: reference not used ({st})")
    for o in sorted(set(r["object"] for r in G)):
      rs = [r for r in G if r["object"] == o]
      views = {r["view"]: r for r in rs}
      P(f"  {o}: {len(rs)} bursts at {len(views)} views")
      P(f"    reference quality: scan points within 1 mm {stats(r.get('gt_fitness_1mm') for r in views.values())}")
      P(f"                       RMSE mm {stats(r.get('gt_rmse_mm') for r in views.values())}")
      P(f"                       margin to next pose {stats(r.get('gt_margin') for r in views.values())}")
      P(f"                       repeat scans, mm {stats(r.get('gt_repeat_mm') for r in views.values())}"
        f"; deg {stats(r.get('gt_repeat_deg') for r in views.values())}")
      acc = [r for r in rs if r["state"] == "Valid"]
      P(f"    accepted poses vs reference: rot deg {stats(r.get('gt_rot_err_deg') for r in acc)}")
      P(f"                                 trans mm {stats(r.get('gt_trans_err_mm') for r in acc)}")
      c = Counter(r["gt_outcome"] for r in rs if r.get("gt_outcome"))
      d = {k: c.get(k, 0) for k in OUTCOMES}
      nacc, nwrong, ngood = d["TA"] + d["FA"] + d["AO"], d["FA"] + d["TR"], d["TA"] + d["FR"]
      P("    gate outcomes graded by the Gocator: " + ", ".join(f"{k} {d[k]}" for k in OUTCOMES if d[k]))
      P(f"    FA rate {pct(d['FA'], nwrong)}   precision {pct(d['TA'], nacc)}   "
        f"FR rate {pct(d['FR'], ngood)}")
      both = [r for r in rs if r.get("grade") in ("correct", "borderline", "wrong")
              and r.get("has_pose") in ("True", True)]
      agree = sum(r["grade"] == r["gt_grade"] for r in both)
      P(f"    your grade = Gocator grade: {pct(agree, len(both))}")
      for (yo, gg), k in sorted(Counter((r["grade"], r["gt_grade"]) for r in both
                                        if r["grade"] != r["gt_grade"]).items()):
        P(f"      you {yo:10s} Gocator {gg:10s} {k}")
      bl = [r for r in rs if r.get("baseline_gt_grade")]
      if bl:
        P(f"    wrong poses reaching the gripper (Gocator-graded): plain RANSAC+ICP "
          f"{sum(r['baseline_gt_grade'] == 'wrong' for r in bl)}/{len(bl)} vs V-GAP "
          f"{sum(r['gt_grade'] == 'wrong' and r['state'] == 'Valid' for r in bl)}/{len(bl)}")

  txt = "\n".join(L)
  with open(out_txt, "w") as f:
    f.write(txt + "\n")
  with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["level", "group", "n", "accepted", "wrong",
                                      "good"] + OUTCOMES)
    w.writeheader()
    w.writerows(summary_rows)
  return txt


# ==============================================================================
# Main
# ==============================================================================
def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--report", nargs="+", default=None,
                  help="only build a combined report from these run folders")
  ap.add_argument("--object", choices=list(vp.OBJECT_CONFIGS))
  ap.add_argument("--color")
  ap.add_argument("--tau-c", type=float, default=None,
                  help="gate threshold to test (default: OBJECT_CONFIGS)")
  ap.add_argument("--axis-ambig-deg", type=float, default=None,
                  help="hemisphere: reject when the axis is ambiguous by more than this "
                       "(deg); config default 10, 15 fixes the propped false rejects")
  ap.add_argument("--cad", default=None, help="CAD file to use instead of the config's")
  ap.add_argument("--cad-units", default="m", choices=["m", "mm"])
  ap.add_argument("--robot", default="dry-run", choices=["dry-run", "ur-rtde", "sim"])
  ap.add_argument("--static", action="store_true",
                  help="camera stays where you put it: no robot connection, no motion. "
                       "--angles is one value, used only as the label (e.g. 60)")
  ap.add_argument("--static-tcp", default=None,
                  help="with --static: TCP pose from the pendant x,y,z,rx,ry,rz (m, "
                       "rotvec), only for the base-frame columns; default identity")
  ap.add_argument("--robot-ip", default="192.168.1.5")
  ap.add_argument("--serial", default=None, help="wrist RealSense serial")
  ap.add_argument("--hand-eye", default=None, help="T_flange_cam.npy (4x4)")
  ap.add_argument("--angles", default="0,30,45",
                  help="tilt from straight down; 60 collided on the real robot")
  ap.add_argument("--side", default="+y", choices=list(tr.SIDES))
  ap.add_argument("--orientations", type=int, default=None)
  ap.add_argument("--reps", type=int, default=10, help="bursts per view")
  ap.add_argument("--placements", default="flat,propped",
                  help="each angle runs all orientations once per placement: "
                       + ", ".join(PLACEMENTS))
  ap.add_argument("--conditions", default="normal",
                  help="comma list from: " + ", ".join(CONDITIONS))
  ap.add_argument("--condition-angles", default="45",
                  help="angles where the extra conditions are run")
  ap.add_argument("--retry", action="store_true", help="test the recovery move")
  ap.add_argument("--retry-step", type=float, default=15.0)
  ap.add_argument("--same-view-retries", type=int, default=1,
                  help="fresh bursts from the same view before the first arm move "
                       "(only when YOLO found the object; 0 = off)")
  ap.add_argument("--retries-per-view", type=int, default=2)
  ap.add_argument("--max-tilt", type=float, default=45.0,
                  help="no view or recovery move tilts more than this (60 collided)")
  ap.add_argument("--retry-orbit", action="store_true",
                  help="if the tilt retry is still not Valid, look from the side "
                       "(same tilt from the table, 90 deg around the object)")
  ap.add_argument("--orbit-tilt", type=float, default=45.0)
  ap.add_argument("--orbit-out", type=float, default=0.0,
                  help="move the orbit views this far (m) farther out from the object, "
                       "same camera height, still aimed at it (bigger tilt); occlusion "
                       "tests: 0.12, so the 90 deg view is not almost top-down")
  ap.add_argument("--retry-best", action="store_true",
                  help="recovery step 1 goes to the object's best angle (--best-angles) "
                       "instead of +/- --retry-step; the orbit then keeps that tilt")
  ap.add_argument("--best-angles", default=BEST_ANGLES_DEFAULT,
                  help="object:deg list; from the 10/22 gate runs, the tilt with the "
                       "most correct accepted poses")
  ap.add_argument("--orbit-step", type=float, default=90.0,
                  help="counter-clockwise (seen from above) deg between orbit stops")
  ap.add_argument("--no-overhead", action="store_true",
                  help="skip the last recovery step (straight above the object)")
  ap.add_argument("--orbit-stops", type=int, default=1,
                  help="orbit stops tried in order, stopping at the first Valid")
  ap.add_argument("--orbit-waypoint", type=float, default=15.0,
                  help="deg between moveL waypoints along the orbit arc")
  ap.add_argument("--plan-only", action="store_true",
                  help="print every view and recovery pose (needs --target), no motion")
  ap.add_argument("--point-check", action="store_true",
                  help="hover the gripper tip over the predicted top center")
  ap.add_argument("--hover", type=float, default=0.03, help="tip gap for --point-check, m")
  ap.add_argument("--dist", type=float, default=0.40)
  ap.add_argument("--speed", type=float, default=0.10)
  ap.add_argument("--tool-tip", default="0,0,0.086")
  ap.add_argument("--table-z", type=float, default=0.0)
  ap.add_argument("--min-cam-height", type=float, default=0.08)
  ap.add_argument("--min-tip-height", type=float, default=0.03)
  ap.add_argument("--min-tcp-height", type=float, default=0.08)
  ap.add_argument("--min-tip-object", type=float, default=0.10)
  ap.add_argument("--min-base-dist", type=float, default=0.20)
  ap.add_argument("--max-reach", type=float, default=0.85)
  ap.add_argument("--target", default=None, help="object center x,y,z (skip finding it)")
  ap.add_argument("--gocator", action="store_true",
                  help="after the bursts at each view, move the wrist Gocator 3210 in "
                       "(same tilt), scan, and grade every pose against the Gocator "
                       "reference (gt_* columns, report section L)")
  ap.add_argument("--gocator-hand-eye", default="data/handeye/T_flange_goc.npy",
                  help="from gocator_handeye.py (the .json next to it gives view axis / standoff)")
  ap.add_argument("--gocator-ip", default="192.168.1.10")
  ap.add_argument("--gocator-dz", type=float, default=0.0,
                  help="object center this far (m) beyond the calibration's average sphere "
                       "spot, along the tool axis (+ = farther from the Gocator)")
  ap.add_argument("--gocator-standoff", type=float, default=None,
                  help="sensor -> object center, m (default: the calibration's standoff)")
  ap.add_argument("--gocator-table-z", type=float, default=0.008,
                  help="table height (base frame, m) for the Gocator reference: the "
                       "table plane comes from the arm pose + this height, refined on "
                       "the table points it sees (0.008 = Gocator hand-eye, 11/01)")
  ap.add_argument("--gocator-find-table", action="store_true",
                  help="old behaviour: search the scan for the table instead")
  ap.add_argument("--gt-min-fitness", type=float, default=0.95,
                  help="Gocator reference with fewer scan points within 1 mm, or repeat "
                       "scans disagreeing by > 2 deg / 2 mm, is marked invalid (not graded)")
  ap.add_argument("--gocator-scans", type=int, default=3,
                  help="scans per view (first: full search; others: repeatability)")
  ap.add_argument("--gripper-roi", default=",".join(map(str, vp.GRIPPER_ROI)),
                  help="x1,y1,x2,y2 px of the gripper corner; YOLO detections mostly "
                       "inside it are dropped")
  ap.add_argument("--no-gripper-mask", action="store_true", help="turn the gripper mask off")
  ap.add_argument("--target-max-dist", type=float, default=vp.TARGET_MAX_DIST_M,
                  help="drop the detection nearest the target if its center is farther "
                       "than this (m, table plane); 0 = no limit")
  ap.add_argument("--no-target-filter", action="store_true",
                  help="pick the most confident detection, not the one nearest the target")
  ap.add_argument("--no-confirm", action="store_true")
  ap.add_argument("--no-inject", action="store_true")
  ap.add_argument("--no-gui", action="store_true")
  ap.add_argument("--demo-video", nargs="?", const="auto", default=None,
                  help="record a split-screen demo video (live RGB + YOLO, 3D check, 2D gate, "
                       "verdict) of the whole run; optional path, default <out>/demo_video.mp4")
  ap.add_argument("--demo-fps", type=float, default=12.0)
  ap.add_argument("--demo-size", default="1920x1080", help="video size WxH")
  ap.add_argument("--demo-no-live-yolo", action="store_true",
                  help="live panel without YOLO masks (lighter on the Jetson)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=None)
  a = ap.parse_args()

  if a.report:
    dirs = [d for p in a.report for d in glob.glob(p) if os.path.isdir(d)]
    out = a.out or os.path.join("data", "gate_eval")
    os.makedirs(out, exist_ok=True)
    print(report(dirs, os.path.join(out, "combined_gate_report.txt"),
                 os.path.join(out, "combined_gate_summary.csv")))
    return
  if not (a.object and a.color):
    raise SystemExit("--object and --color are needed (or --report folders)")

  a.tool_tip = np.asarray(cal.parse_list(a.tool_tip))
  conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
  for c in conds:
    if c not in CONDITIONS:
      raise SystemExit(f"unknown condition {c}; use {list(CONDITIONS)}")
  places = [p.strip() for p in a.placements.split(",") if p.strip()]
  for p in places:
    if p not in PLACEMENTS:
      raise SystemExit(f"unknown placement {p}; use {list(PLACEMENTS)}")
  cond_angles = cal.parse_list(a.condition_angles)
  best_angle = {k.strip(): float(v) for k, v in
                (x.split(":") for x in a.best_angles.split(",") if x.strip())}
  if a.retry_best:
    if a.object not in best_angle:
      raise SystemExit(f"--best-angles has no entry for {a.object}")
    if best_angle[a.object] > 0:   # orbit keeps the best tilt (0 = same view)
      a.orbit_tilt = best_angle[a.object]
  angles = cal.parse_list(a.angles)

  def orbit_geom():
    """(tilt deg, dist m) of the orbit views. --orbit-out pushes the camera
    out horizontally at the same height (10/26 occlusion tests: at the best
    angle, 30 deg, the 90 deg view looked almost straight down past the
    board, so the object was not really occluded there)."""
    h = a.dist * np.cos(np.radians(a.orbit_tilt))
    r = a.dist * np.sin(np.radians(a.orbit_tilt)) + a.orbit_out
    return float(np.degrees(np.arctan2(r, h))), float(np.hypot(r, h))

  def orbit_path(az0):
    """Orbit stops counter-clockwise from az0: (waypoint azimuths from the
    previous stop, stop azimuth, stop number) for each stop."""
    out, prev = [], az0
    for k in range(1, a.orbit_stops + 1):
      az = az0 + k * a.orbit_step
      n = max(1, int(np.ceil((az - prev) / a.orbit_waypoint)))
      out.append(([prev + (az - prev) * i / n for i in range(1, n)], az, k))
      prev = az
    return out

  if a.static:
    if len(angles) != 1:
      raise SystemExit("--static: give one --angles value (the tilt you set by hand)")
    if a.retry or a.point_check or a.plan_only or a.robot == "sim":
      raise SystemExit("--static cannot be used with --retry, --point-check, "
                       "--plan-only or --robot sim (they move the arm)")
    # --gocator is allowed: the Gocator then scans from the same static pose
    # (place the arm so the object is ~22-27 cm from the Gocator glass).
  elif any(t > a.max_tilt for t in angles) or a.orbit_tilt > a.max_tilt:
    raise SystemExit(f"an angle is above --max-tilt {a.max_tilt:g} (60 deg collided)")
  n_ori = a.orientations or tr.ORIENTATIONS[a.object]
  gui = not (a.no_gui or a.robot == "sim")

  if a.hand_eye:
    X = np.load(a.hand_eye).astype(float)
  elif a.robot == "sim":
    X = np.eye(4)
    X[:3, 3] = [0.0, 0.04, -0.107]
  else:
    raise SystemExit("--hand-eye is needed (your T_flange_cam.npy)")

  if a.plan_only:
    if a.target is None:
      raise SystemExit("--plan-only needs --target x,y,z")
    c = np.asarray(cal.parse_list(a.target))
    plan = [(f"view {t:g}", t, a.side) for t in angles]
    if a.retry_best:
      b = best_angle[a.object]
      plan += [(f"best-angle retry {b:g}", b, a.side)]
    else:
      plan += [(f"tilt retry {t + a.retry_step:g}", t + a.retry_step, a.side) for t in angles]
    plan = [(n, t, SIDE_AZ[s_], a.dist) for n, t, s_ in plan]
    o_tilt, o_dist = orbit_geom()
    for _, az, _ in orbit_path(SIDE_AZ[a.side]):
      plan.append((f"orbit ccw {az - SIDE_AZ[a.side]:g} ({o_tilt:.1f} deg)", o_tilt, az, o_dist))
    if not a.no_overhead:
      plan.append(("overhead 0", 0.0, SIDE_AZ[a.side], a.dist))
    if a.same_view_retries:
      print(f"  (on a Conflict: {a.same_view_retries} fresh burst(s) at the same view, "
            "no motion, before the recovery moves below)")
    for name, t, az, d in plan:
      T = view_pose_az(c, t, az, d, X)
      try:
        if t > a.max_tilt:
          raise ValueError(f"tilt {t:.1f} deg above --max-tilt {a.max_tilt:g}")
        tr.check_view(T, X, c, a)
        print(f"  {name:18s} TCP {np.round(cal.to_ur_pose(T), 4).tolist()}")
      except ValueError as e:
        print(f"  {name:18s} SKIPPED ({e})")
    return

  if a.robot == "sim":
    import sim_stress_test as st
    for k, rel in st.MESHES.items():
      vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(st.ROOT, rel)
      vp.OBJECT_CONFIGS[k]["cad_units"] = "mm"
    cam = cal.SimCamera(a.object, a.color, a.seed, X)
    robot = cal.DryRunRobot(False, cal.from_ur_pose(tr.OBSERVE_POSE))
    if conds != ["normal"] or places != ["flat"]:
      print("[sim] only the flat placement and 'normal' condition can be simulated")
      conds, places = ["normal"], ["flat"]
  else:
    cam = cal.RealSenseCamera(a.serial)
    cam.gripper_roi = None if a.no_gripper_mask else \
        tuple(int(v) for v in a.gripper_roi.split(","))
    cam.max_dist = a.target_max_dist
    if a.static:
      robot = StaticRobot(cal.from_ur_pose(cal.parse_list(a.static_tcp))
                          if a.static_tcp else np.eye(4))
      print("[static] no robot connection; the arm is never moved")
    elif a.robot == "ur-rtde":
      robot = cal.URRtdeRobot(a.robot_ip, speed=a.speed, acc=0.2,
                              confirm=not a.no_confirm)
    else:
      robot = cal.DryRunRobot(wait_for_operator=True,
                              start_pose=cal.from_ur_pose(tr.OBSERVE_POSE))
  params = vp.OBJECT_CONFIGS[a.object]
  if a.cad and a.robot != "sim":
    params["cad_path"], params["cad_units"] = a.cad, a.cad_units
  if a.tau_c is not None:
    params["tau_c"] = a.tau_c
  if a.axis_ambig_deg is not None:
    params["axis_ambig_deg"] = a.axis_ambig_deg
  half_h = min(params["expected_extent_mm"]) / 2000.0

  cad_cache = {}
  for k in [a.object] + vp.CLASS_RIVALS.get(a.object, []):
    with contextlib.redirect_stdout(io.StringIO()) if cam.simulated else \
        contextlib.nullcontext():
      cad_cache[k] = vp.load_cad_model(vp.OBJECT_CONFIGS[k])
  cad = cad_cache[a.object]

  gcam = None
  if a.gocator:
    if a.robot != "ur-rtde" and not a.static:
      raise SystemExit("--gocator needs --robot ur-rtde (the arm moves the Gocator in)")
    import gocator_gt as ggt
    from gocator_cam import GocatorCamera
    Xg = np.load(a.gocator_hand_eye).astype(float)
    with open(a.gocator_hand_eye.replace(".npy", ".json")) as f:
      g_info = json.load(f)
    if "center_gocator_m" in g_info:
      # put the object where the hand-eye calibration saw the sphere on average,
      # shifted --gocator-dz along the tool axis (+ = farther from the Gocator)
      g_axis = np.asarray(g_info["tool_axis_gocator"], float)
      g_spot = np.asarray(g_info["center_gocator_m"], float) + a.gocator_dz * g_axis
      g_standoff = 0.22                      # nominal glass -> spot, only for clearance checks
    else:                                    # old calibration file
      g_axis = np.asarray(g_info["view_axis_gocator"], float)
      g_standoff = a.gocator_standoff or g_info["standoff_m"]
      g_spot = g_standoff * g_axis
    g_target = ggt.CadTarget(cad, vp.OBJECT_CONFIGS[a.object])
    gcam = GocatorCamera(a.gocator_ip)
    print(f"[+] Gocator reference on: spot {np.round(1000 * g_spot, 1)} mm (Gocator frame), hand-eye "
          f"residual {g_info['residual_mean_mm']} mm")

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = a.out or os.path.join("data", "gate_eval", f"{a.object}_{a.color}_{stamp}")
  os.makedirs(out_dir, exist_ok=True)
  files = {k: open(os.path.join(out_dir, f"{k}.csv"), "w", newline="")
           for k in ("gate_eval", "injected", "recovery", "point_check", "gocator")}
  writers = {}

  def write(kind, row):
    if kind not in writers:
      writers[kind] = csv.DictWriter(files[kind], fieldnames=list(row.keys()),
                                     extrasaction="ignore")
      writers[kind].writeheader()
    writers[kind].writerow(row)
    files[kind].flush()

  def sim_grade(T_bt, out):
    if not cam.simulated:
      return None
    if "T_cad_to_cam" not in out or out.get("final_object") != a.object:
      return "wrong"
    return cal.label_of(*cal.pose_error(cam.true_pose_cam(T_bt), out["T_cad_to_cam"], cad))

  def find_object(from_pose, label):
    robot.move_tcp(from_pose, label)
    T_bt = robot.get_tcp()
    obs = cam.capture(T_bt)
    return None if obs is None else tr.locate(obs, T_bt @ X, half_h)

  def sim_rotate_object():
    c = cam.object_center()
    T = np.eye(4)
    T[:3, :3] = vp._rot([0, 0, 1], 90.0)
    T[:3, 3] = c - T[:3, :3] @ c
    cam.T_base_obj = T @ cam.T_base_obj

  demo = None                           # --demo-video recorder (set below)

  def burst(T_bt):
    if demo is not None:
      demo.set_status("capturing burst")
    obs = cam.capture(T_bt)
    if obs is None:
      if demo is not None:
        demo.set_status("")
        demo.on_burst(None, None, None)
      return None, None
    _BASELINE.clear()
    if demo is not None:
      demo.set_status("registering: RANSAC + ICP, then the gate")
    out = cal.run_pipeline(obs, cad_cache, cam.simulated)
    Tb = baseline_pose(obs, cad_cache)
    if Tb is not None:
      out["baseline_T_c2c"] = Tb
    if demo is not None:
      demo.set_status("")
      with contextlib.suppress(Exception):
        demo.on_burst(obs, out, cad_cache.get(out.get("final_object", obs.object_key)))
    return obs, out

  def burst_row(base, obs, out, T_bt):
    filt = {**DET_FILTER_EMPTY, **(getattr(cam, "last_filter", None) or {})}
    if obs is None:
      return dict(base, state="no detection", reason="no detection", has_pose=False,
                  **filt)
    fin = out.get("final_object", obs.object_key)
    r = dict(base, **filt, yolo_object=obs.object_key, yolo_conf=round(obs.confidence, 3),
             final_object=fin, class_swapped=out.get("class_swapped", False),
             class_margin=out.get("class_margin", ""),
             class_scores=json.dumps(out.get("class_scores", {})),
             state=out["state"], reason=out.get("reason", ""),
             C=round(out.get("C", 0.0), 4), IB=round(out.get("IB", 0.0), 4),
             IM=round(out.get("IM", 0.0), 4),
             margin=round(out.get("margin", float("nan")), 4),
             UD=round(out.get("UD", float("nan")), 4),
             U_live=round(out.get("U_live", float("nan")), 4),
             used_fallback=out.get("used_fallback", False), chosen=out.get("chosen", ""),
             axis_ambiguity_deg=out.get("axis_ambiguity_deg"),
             runtime_s=round(out.get("runtime_s", float("nan")), 2),
             mask_px=int(obs.mask.sum()), has_pose="T_cad_to_cam" in out,
             hypothesis_used=out.get("chosen", "as registered") not in ("", "as registered"))
    if "baseline_T_c2c" in out:
      bc = cad_cache[obs.object_key]
      _, _, Cb = cal.score_pose(obs, bc, out["baseline_T_c2c"], vp.OBJECT_CONFIGS[obs.object_key])
      r["baseline_C"] = round(Cb, 4)
    if "T_cad_to_cam" in out:
      Tcc = out["T_cad_to_cam"]
      r.update(tr.metrics_3d(out.get("scan"), cad_cache[fin], Tcc))
      r.update(tr.pose_fields(Tcc, "cam"))
      r.update(tr.pose_fields(T_bt @ X @ Tcc, "base"))
      if cam.simulated:
        e_r, e_t = cal.pose_error(cam.true_pose_cam(T_bt), Tcc, cad)
        r.update(true_rot_err_deg=round(e_r, 2), true_trans_err_mm=round(e_t, 2))
      else:
        d = os.path.join(out_dir, "trials")
        obs.meta["timestamp"] = f"{base['view']}_r{base['rep']}_{obs.meta['timestamp']}"
        with contextlib.redirect_stdout(io.StringIO()):
          vp.save_trial(obs, out, root=d)
        r["trial_dir"] = os.path.join(d, f"{obs.object_key}_{obs.meta['timestamp']}")
    return r

  ROW_KEYS = None

  def tilt_retry_pose(center, ang):
    if a.retry_best:
      b = best_angle[a.object]
      if abs(b - ang) < 1e-6 or b > a.max_tilt:
        return None              # already at the best angle: go straight to the orbit
      T = tr.view_pose(center, b, a.side, a.dist, X)
      try:
        tr.check_view(T, X, center, a)
        return T, b
      except ValueError:
        return None
    for s_ in (a.retry_step, -a.retry_step):
      if ang + s_ < 0 or ang + s_ > a.max_tilt:
        continue
      T = tr.view_pose(center, ang + s_, a.side, a.dist, X)
      try:
        tr.check_view(T, X, center, a)
        return T, ang + s_
      except ValueError:
        continue
    return None

  def gocator_view(view, T_view, place):
    """Move the Gocator in at the same tilt, scan, reference pose (base frame).
    --static: no move, the Gocator scans from where the arm already is. V-GAP and
    the reference then share the same (possibly unknown) TCP pose, so the
    comparison is still exact."""
    if a.static:
      return gocator_scan(view, place)
    R = T_view[:3, :3]
    T_g = np.eye(4)
    T_g[:3, :3] = R
    T_g[:3, 3] = center - R @ (Xg[:3, :3] @ g_spot + Xg[:3, 3])
    Xg_glass = Xg.copy()                     # approx. sensor glass, for the clearance check
    Xg_glass[:3, 3] = Xg[:3, :3] @ (g_spot - g_standoff * g_axis) + Xg[:3, 3]
    try:
      tr.check_view(T_g, Xg_glass, center, a)
    except ValueError as e:
      print(f"  [gocator] view unsafe, no reference here: {e}")
      return dict(gt_status=f"unsafe: {e}")
    robot.move_tcp(T_g, "Gocator in")
    res = gocator_scan(view, place)
    robot.move_tcp(T_view, f"back to {view}")
    return res

  def gocator_scan(view, place):
    res = dict(gt_status="ok")
    try:
      T_bt_g = robot.get_tcp()
      T_s_b = np.linalg.inv(T_bt_g @ Xg)
      near = T_s_b[:3, :3] @ center + T_s_b[:3, 3] if center is not None else None
      gdir = os.path.join(out_dir, "gocator")
      os.makedirs(gdir, exist_ok=True)
      Ts = []
      for k in range(max(1, a.gocator_scans)):
        pts = gcam.snapshot()
        np.save(os.path.join(gdir, f"{view}_s{k + 1}.npy"), pts.astype(np.float32))
        tbl = None if a.gocator_find_table else ggt.table_in_sensor(T_bt_g @ Xg, a.gocator_table_z)
        obj, plane = ggt.isolate_object(pts, near=near, view_dir=g_axis, table=tbl)
        if k == 0:
          g = ggt.gt_pose(obj, g_target, plane)
          T0 = g["T_cad_to_sensor"]
          Ts.append(T0)
          res.update(gt_fitness_1mm=round(g["fitness_1mm"], 4),
                     gt_rmse_mm=round(g["rmse_mm"], 3),
                     gt_margin=round(g["margin"], 4) if g["margin"] == g["margin"] else "",
                     gt_n_points=g["n_points"])
          rests = g.get("rests_on_table", True)
        else:
          Tk, _, _ = ggt.gt_refine(obj, g_target, T0)
          Ts.append(Tk)
      spr = [cal.pose_error(Ts[0], T, cad) for T in Ts[1:]]
      if spr:
        res.update(gt_repeat_deg=round(max(e[0] for e in spr), 3),
                   gt_repeat_mm=round(max(e[1] for e in spr), 3))
      res["T_base_obj"] = T_bt_g @ Xg @ T0
      res["gt_rot_valid"] = not (cad.axial_axis is not None and place != "flat")
      bad = []
      if not rests:
        bad.append("no fit rests on the table (part not fully seen)")
      if res.get("gt_fitness_1mm", 0) < a.gt_min_fitness:
        bad.append(f"fitness {res.get('gt_fitness_1mm')}")
      if spr and (res["gt_repeat_deg"] > 2 or res["gt_repeat_mm"] > 2):
        bad.append(f"repeat {res['gt_repeat_deg']} deg / {res['gt_repeat_mm']} mm")
      if bad:
        res["gt_status"] = "invalid: " + ", ".join(bad)
        print(f"  [gocator] reference INVALID ({', '.join(bad)}): not used for grading")
      write("gocator", dict(object=a.object, color=a.color, view=view,
                            gt_tcp=json.dumps([round(v, 6) for v in cal.to_ur_pose(T_bt_g)]),
                            scans=len(Ts), **{k: v for k, v in res.items()
                                              if k != "T_base_obj"},
                            gt_base=json.dumps(np.round(res["T_base_obj"], 6).tolist())))
    except Exception as e:
      print(f"  [gocator] failed: {e}")
      res = dict(gt_status=f"failed: {e}")
    if res.get("gt_status") != "ok":
      res.pop("T_base_obj", None)   # invalid reference: never grade against it
    return res

  def gt_fields(gt, T_bt, out, prefix="gt_"):
    """V-GAP (or baseline) pose vs the Gocator reference."""
    if not gt or "T_base_obj" not in gt:
      return {}
    key = "T_cad_to_cam" if prefix == "gt_" else "baseline_T_c2c"
    if out is None or key not in out:
      return {f"{prefix}grade": "wrong"}
    fin = out.get("final_object", a.object) if prefix == "gt_" else a.object
    T_b = T_bt @ X @ out[key]
    e_r, e_t = cal.pose_error(gt["T_base_obj"], T_b, cad)
    rot_ok = gt.get("gt_rot_valid", True)
    grade = cal.label_of(e_r if rot_ok else 0.0, e_t, wrong_class=fin != a.object)
    return {f"{prefix}rot_err_deg": round(e_r, 2) if rot_ok else "",
            f"{prefix}trans_err_mm": round(e_t, 2), f"{prefix}grade": grade}

  def run_view(ang, oi, cond, T_view, place="flat"):
    """10 bursts, grading, outcomes, then the optional tests at this view."""
    nonlocal ROW_KEYS
    view = f"a{ang:g}_{place}_o{oi + 1}_{cond}"
    group = []
    for rep in range(a.reps):
      if demo is not None:
        demo.set_stage(f"First view {ang:g} deg ({place}, {cond}): burst {rep + 1}/{a.reps}")
      T_bt = robot.get_tcp()
      obs, out = burst(T_bt)
      base = dict(object=a.object, color=a.color, tau_c=params["tau_c"],
                  angle_deg=ang, placement=place, orientation=oi + 1,
                  condition=cond, view=view,
                  rep=rep + 1, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                  tcp_pose=json.dumps([round(v, 6) for v in cal.to_ur_pose(T_bt)]))
      r = burst_row(base, obs, out, T_bt)
      group.append(dict(row=r, obs=obs, out=out, T_bt=T_bt))
      print(f"  [{view} / {rep + 1}] {r.get('final_object', '-')}: {r['state']} "
            f"C={_f(r.get('C')) or 0:.3f} {r.get('reason', '')}")

    gt = gocator_view(view, T_view, place) if gcam is not None else None

    # ---- grade each distinct pose once
    posed = [g for g in group if g["out"] is not None and "T_cad_to_cam" in g["out"]]
    T_ref = None
    if posed:
      Ts = [g["out"]["T_cad_to_cam"] for g in posed]
      med = np.median([T[:3, 3] for T in Ts], axis=0)
      i_med = int(np.argmin([np.linalg.norm(T[:3, 3] - med) for T in Ts]))
      for g, T in zip(posed, Ts):
        e_r, e_t = cal.pose_error(Ts[i_med], T, cad)
        g["row"].update(dev_from_median_mm=round(e_t, 2), dev_from_median_deg=round(e_r, 2))
      keyed = [(g["out"].get("final_object"), g) for g in posed]
      lab = []
      reps_ = []
      for fin, g in keyed:       # cluster per class
        for k, (f2, T2) in enumerate(reps_):
          if f2 == fin:
            e_r, e_t = cal.pose_error(T2, g["out"]["T_cad_to_cam"], cad)
            if e_r <= 5 and e_t <= 5:
              lab.append(k)
              break
        else:
          reps_.append((fin, g["out"]["T_cad_to_cam"]))
          lab.append(len(reps_) - 1)
      grades = {}
      for k in sorted(set(lab), key=lambda k: -lab.count(k)):
        g = posed[lab.index(k)]
        if g["out"].get("final_object") != a.object:
          grades[k] = "wrong"
          print(f"  pose group {k + 1}: {lab.count(k)} bursts, wrong class "
                f"({g['out'].get('final_object')}) -> wrong")
          continue
        print(f"  pose group {k + 1}: {lab.count(k)} of {len(posed)} bursts, "
              f"C={g['row']['C']:.3f} {g['row']['state']}")
        grades[k] = ask_grade(g["obs"], g["out"], gui, sim_grade(g["T_bt"], g["out"]),
                              f"group {k + 1}")
        if grades[k] == "correct" and T_ref is None:
          T_ref = g["out"]["T_cad_to_cam"]
      for g, k in zip(posed, lab):
        g["row"]["grade"] = grades[k]
        g["row"]["grade_source"] = "you" if not cam.simulated else "sim truth"
        g["row"]["pose_group"] = k + 1
    # ---- plain RANSAC+ICP baseline: same grade when it is the same pose,
    #      otherwise you grade it too (once per distinct baseline pose)
    need = []
    for g in group:
      r, out = g["row"], g["out"]
      Tb = out.get("baseline_T_c2c") if out else None
      if Tb is None:
        continue
      if g["obs"].object_key != a.object:
        r.update(baseline_grade="wrong", baseline_same_as_final=False)
        continue
      if "T_cad_to_cam" in out and out.get("final_object") == a.object:
        e_r, e_t = cal.pose_error(out["T_cad_to_cam"], Tb, cad)
        r.update(baseline_diff_deg=round(e_r, 2), baseline_diff_mm=round(e_t, 2))
        if e_r <= 5 and e_t <= 5:
          r.update(baseline_grade=r.get("grade", "wrong"), baseline_same_as_final=True)
          continue
      r["baseline_same_as_final"] = False
      need.append(g)
    reps_b = []
    for g in need:
      Tb = g["out"]["baseline_T_c2c"]
      for Tr, grade in reps_b:
        e_r, e_t = cal.pose_error(Tr, Tb, cad)
        if e_r <= 5 and e_t <= 5:
          g["row"]["baseline_grade"] = grade
          break
      else:
        Mp, _ = vp.project_silhouette(cad.dense_points, Tb, g["obs"].K, g["obs"].mask.shape)
        fake = dict(Mproj=Mp, C=g["row"].get("baseline_C", 0.0),
                    state="plain RANSAC+ICP", final_object=a.object)
        sg = None
        if cam.simulated:
          sg = cal.label_of(*cal.pose_error(cam.true_pose_cam(g["T_bt"]), Tb, cad))
        print("  plain RANSAC+ICP gave a different pose than V-GAP's final one:")
        grade = ask_grade(g["obs"], fake, gui, sg, "plain RANSAC+ICP")
        reps_b.append((Tb, grade))
        g["row"]["baseline_grade"] = grade
    for g in group:
      r = g["row"]
      if "baseline_grade" in r:
        b, f = r["baseline_grade"], r.get("grade", "wrong")
        r["vgap_effect"] = ("rescued" if b != "correct" and f == "correct" else
                            "harmed" if b == "correct" and f != "correct" else "same")
        r["changed_by"] = ("class check" if r.get("class_swapped") else
                           "fallback" if r.get("used_fallback") else
                           "hypothesis" if r.get("hypothesis_used") else
                           "" if r.get("baseline_same_as_final") else "other")

    for g in group:
      r = g["row"]
      if "grade" not in r:
        r["grade"], r["grade_source"] = "wrong", "no pose"
      if T_ref is not None and r.get("has_pose"):
        e_r, e_t = cal.pose_error(T_ref, g["out"]["T_cad_to_cam"], cad)
        r.update(dev_ref_mm=round(e_t, 2), dev_ref_deg=round(e_r, 2))
      r["outcome"] = outcome(r["state"], r["grade"])
      if gt is not None:
        r.update({k: v for k, v in gt.items() if k != "T_base_obj"})
        r.update(gt_fields(gt, g["T_bt"], g["out"]))
        r.update(gt_fields(gt, g["T_bt"], g["out"], "baseline_gt_"))
        if "gt_grade" in r:
          r["gt_outcome"] = outcome(r["state"], r["gt_grade"])
      if ROW_KEYS is None:
        ROW_KEYS = list(dict.fromkeys(list(r.keys()) + [
            "fitness_3mm", "rmse_mm", "within_3mm", "within_5mm", "median_dist_mm",
            "n_scan_points", "dev_from_median_mm", "dev_from_median_deg", "dev_ref_mm",
            "dev_ref_deg", "pose_group", "true_rot_err_deg", "true_trans_err_mm",
            "trial_dir"] + list(DET_FILTER_EMPTY) + GT_KEYS + ["baseline_C", "baseline_diff_deg", "baseline_diff_mm",
            "baseline_same_as_final", "baseline_grade", "vgap_effect", "changed_by"]
            + [f"{p}_{s}" for p in ("cam", "base")
                            for s in ("x_mm", "y_mm", "z_mm", "rx", "ry", "rz")]))
        writers["gate_eval"] = csv.DictWriter(files["gate_eval"], fieldnames=ROW_KEYS,
                                              extrasaction="ignore")
        writers["gate_eval"].writeheader()
      write("gate_eval", r)
    c = Counter(g["row"]["outcome"] for g in group)
    print(f"  => {view}: " + ", ".join(f"{k} {c[k]}" for k in OUTCOMES if c[k]))

    # ---- E/F adversarial poses around the confirmed pose
    if T_ref is not None and not a.no_inject and cond == "normal":
      g = min(posed, key=lambda g: cal.pose_error(T_ref, g["out"]["T_cad_to_cam"], cad)[1])
      for ir in injected_eval(g["obs"], g["out"].get("scan"), cad, params, T_ref):
        write("injected", dict(object=a.object, color=a.color, tau_c=params["tau_c"],
                               angle_deg=ang, placement=place, orientation=oi + 1,
                               view=view, **ir))

    # ---- H recovery: 0) same view again (object found), 1) best angle (or tilt
    #      step), 2) 90 deg CCW, 3) straight above
    if a.retry:
      tries = [g for g in group if g["row"]["state"] != "Valid"]
      for ci, g in enumerate(tries[:a.retries_per_view]):
        cid = f"{view}_c{ci + 1}"
        stages = []
        if g["row"]["state"] != "no detection":
          stages += [("same", T_view, ang, a.side)] * a.same_view_retries
        T_alt = tilt_retry_pose(center, ang)
        if T_alt is not None:
          stages.append(("tilt", T_alt[0], T_alt[1], a.side))
        if a.retry_orbit:
          # counter-clockwise around the object at the orbit tilt; every
          # waypoint and stop must pass the clearance checks, else the orbit
          # ends there (the arc is never cut short with a straight move)
          az0 = SIDE_AZ[a.side]
          for wps, az, k in orbit_path(az0):
            try:
              o_tilt, o_dist = orbit_geom()
              if o_tilt > a.max_tilt:
                raise ValueError(f"tilt {o_tilt:.1f} deg above --max-tilt {a.max_tilt:g}")
              path = [view_pose_az(center, o_tilt, w, o_dist, X) for w in wps]
              T = view_pose_az(center, o_tilt, az, o_dist, X)
              for W in path + [T]:
                tr.check_view(W, X, center, a)
            except ValueError as e:
              print(f"  [orbit ccw {az - az0:g}] unsafe, orbit ends here: {e}")
              break
            stages.append(("orbit", T, round(o_tilt, 1), f"ccw{az - az0:g}", path))
        # last try straight above, unless 0 deg was already seen (first view
        # or the best-angle step): an occluder may block both side views
        seen_top = ang == 0 or (T_alt is not None and T_alt[1] == 0)
        if a.retry_orbit and not a.no_overhead and not seen_top:
          T = view_pose_az(center, 0.0, SIDE_AZ[a.side], a.dist, X)
          try:
            tr.check_view(T, X, center, a)
            stages.append(("overhead", T, 0.0, "above"))
          except ValueError as e:
            print(f"  [overhead] unsafe: {e}")
        if not stages:
          print("  [retry] no safe recovery view")
          break
        trail = []                      # poses passed on the orbit, to go back the same way
        for si, (stage, T, alt, side, *rest) in enumerate(stages):
          if demo is not None:
            demo.set_stage({"same": "Recovery step 0: same view again",
                            "tilt": f"Recovery step 1: best angle {alt:g} deg",
                            "orbit": f"Recovery step 2: orbit {side} at {alt:g} deg",
                            "overhead": "Recovery step 3: straight overhead"}.get(stage, stage))
          if stage == "orbit" and not trail:
            # first orbit stop: change tilt in the current plane first
            o_tilt, o_dist = orbit_geom()
            T0 = view_pose_az(center, o_tilt, SIDE_AZ[a.side], o_dist, X)
            tr.check_view(T0, X, center, a)
            robot.move_tcp(T0, f"recovery: orbit start {o_tilt:.1f} deg")
            trail.append(T0)
          if stage == "overhead":
            trail = []                  # straight down from here; no arc to retrace
          for W in (rest[0] if rest else []):
            robot.move_tcp(W, "recovery: orbit waypoint")
            trail.append(W)
          if stage != "same":           # same view: new burst, no motion
            robot.move_tcp(T, f"recovery {stage}: {alt:g} deg, side {side}")
          if stage == "orbit":
            trail.append(T)
          T_bt = robot.get_tcp()
          obs2, out2 = burst(T_bt)
          st2 = "no detection" if obs2 is None else out2["state"]
          gr2 = "wrong"
          if obs2 is not None and "T_cad_to_cam" in out2 and \
              out2.get("final_object") == a.object:
            gr2 = ask_grade(obs2, out2, gui, sim_grade(T_bt, out2), f"recovery {stage}")
          oc = outcome(st2, gr2)
          last = st2 == "Valid" or si == len(stages) - 1
          write("recovery", dict(object=a.object, color=a.color, view=view,
                                 placement=place, condition=cond, conflict_id=cid,
                                 stage=stage, attempt=si + 1, side=side,
                                 from_angle=ang, to_angle=alt,
                                 old_state=g["row"]["state"], old_reason=g["row"]["reason"],
                                 old_grade=g["row"]["grade"], old_C=g["row"].get("C"),
                                 new_state=st2,
                                 new_C=round(out2.get("C", 0), 4) if out2 else "",
                                 new_reason=out2.get("reason", "") if out2 else "",
                                 new_grade=gr2, new_outcome=oc, final_attempt=last,
                                 **{("new_" + k): v for k, v in
                                    gt_fields(gt, T_bt, out2 if obs2 is not None else None).items()}))
          print(f"  [recovery {stage} {alt:g} deg {side}] {st2} grade {gr2}")
          if st2 == "Valid":
            break
        for W in reversed(trail[:-1]):  # back along the arc (clockwise)
          robot.move_tcp(W, "recovery: orbit back")
        robot.move_tcp(T_view, f"back to {view}")

    # ---- I physical accuracy (gripper tip over the predicted top center)
    if a.point_check and cond == "normal":
      ok = [g for g in group if g["row"]["outcome"] == "TA"]
      if ok:
        g = ok[0]
        T_bo = g["T_bt"] @ X @ g["out"]["T_cad_to_cam"]
        P = np.asarray(cad.pcd.points) @ T_bo[:3, :3].T + T_bo[:3, 3]
        top = np.array([P[:, 0].mean(), P[:, 1].mean(), P[:, 2].max()])
        R = cal.from_ur_pose([0, 0, 0, np.pi, 0, 0])[:3, :3]
        tip_goal = top + [0, 0, a.hover]
        T_h = np.eye(4)
        T_h[:3, :3], T_h[:3, 3] = R, tip_goal - R @ a.tool_tip
        tcp = T_h[:3, 3]
        if np.linalg.norm(tcp[:2]) > a.max_reach or tcp[2] < a.table_z + a.min_tcp_height:
          print("  [point check] hover pose out of the safe range, skipped")
        else:
          above = T_h.copy()
          above[2, 3] += 0.10
          robot.move_tcp(above, "point check: 10 cm above the predicted top")
          robot.move_tcp(T_h, f"point check: tip {1000 * a.hover:.0f} mm above the top")
          if cam.simulated:
            T_true = g["T_bt"] @ X @ cam.true_pose_cam(g["T_bt"])
            Pt = np.asarray(cad.pcd.points) @ T_true[:3, :3].T + T_true[:3, 3]
            true_top = np.array([Pt[:, 0].mean(), Pt[:, 1].mean(), Pt[:, 2].max()])
            xy = 1000 * np.linalg.norm(top[:2] - true_top[:2])
            gap = 1000 * (tip_goal[2] - true_top[2])
          else:
            print(f"  Measure with a ruler: the tip should be {1000 * a.hover:.0f} mm above "
                  "the marked top center.")
            xy = float(input("    horizontal distance tip -> marked center, mm: "))
            gap = float(input("    gap tip -> top surface, mm: "))
          write("point_check", dict(object=a.object, color=a.color, view=view,
                                    placement=place,
                                    C=g["row"]["C"], xy_err_mm=round(xy, 1),
                                    gap_mm=round(gap, 1),
                                    z_err_mm=round(gap - 1000 * a.hover, 1),
                                    pred_top_m=json.dumps(np.round(top, 4).tolist())))
          print(f"  [point check] horizontal {xy:.1f} mm, gap {gap:.1f} mm")
          robot.move_tcp(above, "point check: back up")
          robot.move_tcp(T_view, f"back to {view}")

  ori_state = dict(now=0, first=True)

  def angle_loop(ang, T_view):
    """All placements x orientations x conditions at one view."""
    ori_now, first = ori_state["now"], ori_state["first"]
    for place in places:
      for oi in range(n_ori):
        turns = (oi - ori_now) % n_ori if n_ori > 1 else 0
        if not first:
          if oi == 0:
            msg = (f"Angle {ang:g}: put the {a.object} {PLACEMENTS[place]}, "
                   f"orientation 1/{n_ori}; {CONDITIONS['normal']}")
          elif turns:
            msg = (f"Angle {ang:g}, {place}: orientation {oi + 1}/{n_ori}: turn it "
                   f"{90 * turns} deg about the vertical, same spot")
          else:
            msg = f"Angle {ang:g}, {place}: orientation {oi + 1}: check it is in place"
          if cam.simulated:
            for _ in range(turns):
              sim_rotate_object()
          tr.wait_for_operator(cam, msg, gui)
        elif place != "flat":
          tr.wait_for_operator(cam, f"Put the {a.object} {PLACEMENTS[place]}", gui)
        first = False
        ori_now = oi
        if "normal" in conds:
          run_view(ang, oi, "normal", T_view, place)
        if ang in cond_angles:
          for cond in conds:
            if cond == "normal":
              continue
            tr.wait_for_operator(cam, f"Angle {ang:g}, {place}, orientation {oi + 1}: "
                                 f"{CONDITIONS[cond]}", gui)
            run_view(ang, oi, cond, T_view, place)
    ori_state["now"], ori_state["first"] = ori_now, first

  if a.demo_video:
    if cam.simulated:
      print("[demo] --demo-video needs the real camera; not recording")
    else:
      import demo_recorder
      vw, vh = (int(v) for v in a.demo_size.lower().split("x"))
      demo = demo_recorder.DemoRecorder(
          os.path.join(out_dir, "demo_video.mp4") if a.demo_video == "auto" else a.demo_video,
          cam, robot, a.object, fps=a.demo_fps, size=(vw, vh),
          live_yolo=not a.demo_no_live_yolo, tau_c=params["tau_c"],
          f3_min=params.get("f3_min") if vp.CHECK_3D else None)
      demo.set_stage("Finding the object")

  try:
    if a.static:
      # no motion, so no search: the target filter runs only with --target
      # (and then needs --static-tcp so both are in the base frame)
      center = np.asarray(cal.parse_list(a.target)) if a.target else None
    elif a.target:
      center = np.asarray(cal.parse_list(a.target))
    else:
      print("[*] Moving to OBSERVE_POSE to find the object")
      center = find_object(cal.from_ur_pose(tr.OBSERVE_POSE), "observe pose")
      if center is None:
        raise SystemExit("No object found from OBSERVE_POSE; pass --target x,y,z")
    if not cam.simulated and not a.no_target_filter and center is not None:
      cam.target, cam.X = center, X     # keep the detection nearest the object
    if center is not None:
      print(f"[+] Object at {np.round(center, 3).tolist()} (base frame), tau_c "
            f"{params['tau_c']}")

    for ang in angles:
      if a.static:                      # camera already where you put it
        T_view = robot.get_tcp()
        print(f"[static] views labeled {ang:g} deg, tau_c {params['tau_c']}")
        angle_loop(ang, T_view)
        continue
      above = tr.view_pose(center, 0.0, a.side, a.dist, X)
      c2 = find_object(above, f"above object (angle {ang:g} next)")
      # Old code (used for the full 1,440-burst gate runs): took the whole
      # re-found center, height included.
      # if c2 is not None and np.linalg.norm(c2[:2] - center[:2]) < 0.10:
      #   center = c2
      #
      # Occlusion tests (10/26): with --target, only x/y come from the re-found
      # center and z stays at --target. All objects use the gearbox's center
      # height (0.146), so for L/T/hemisphere the views aim ~2 cm above their
      # real center. Any lower and the UR5e runs into the poster board used as
      # the occluder (at the gearbox height the arm just grazes over it).
      if c2 is not None and np.linalg.norm(c2[:2] - center[:2]) < 0.10:
        if a.target:
          c2[2] = center[2]
        center = c2
      T_view = tr.view_pose(center, ang, a.side, a.dist, X)
      try:
        tr.check_view(T_view, X, center, a)
      except ValueError as e:
        print(f"[-] angle {ang:g} skipped: {e}")
        continue
      robot.move_tcp(T_view, f"angle {ang:g} deg")
      angle_loop(ang, T_view)
  except KeyboardInterrupt:
    print("\n[!] Stopped; everything so far is saved.")
  finally:
    for f in files.values():
      f.close()
    with contextlib.suppress(Exception):
      if not cam.simulated and not a.static:
        robot.move_tcp(cal.from_ur_pose(tr.OBSERVE_POSE), "back to OBSERVE_POSE")
    robot.close()
    if demo is not None:
      with contextlib.suppress(Exception):
        demo.close()
    cam.close()
    if gcam is not None:
      gcam.close()
  if os.path.getsize(os.path.join(out_dir, "gate_eval.csv")):
    print("\n" + report([out_dir], os.path.join(out_dir, "gate_report.txt"),
                        os.path.join(out_dir, "gate_summary.csv")))
  print(f"\n[+] Saved to {out_dir}\n    All runs together: python vgap_gate_eval_ur5e.py "
        f"--report data/gate_eval/*/")


if __name__ == "__main__":
  main()
