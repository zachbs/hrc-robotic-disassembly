"""Recommends the gate threshold tau_c per object from labelled C values.

Input: one or more calibration CSVs written by calibrate_tau_c_ur5e.py
(real captures), and/or the simulation results with --sim.

Which wrong poses count: real pipeline outputs, plus injected errors AFTER
ICP (the local minima the pipeline can actually return). Raw injected
errors are not what the pipeline outputs (ICP removes most of them), so
they only fill the sensitivity table: at the chosen tau_c, how often is a
15 deg / 20 mm / ... error rejected. --use-raw adds the raw in-plane ones
to the threshold too (stricter). Out-of-plane errors (tilt, depth shift)
barely change the silhouette; the table shows how blind C is to them.

Rule (per object):
  1. zero false accepts: tau_c sits just above the highest C of any WRONG pose
     that no other check would catch (margin < delta, class margin < CLASS_DELTA,
     hemisphere axis ambiguity). Those other checks already turn such poses
     into Conflicts, so they should not force tau_c up.
  2. with that, as many CORRECT poses accepted as possible (it is the lowest
     threshold with zero false accepts, so true accepts are maximal).
Correct = right class, <= 5 deg and <= 5 mm (symmetry-aware). Wrong = wrong
class, > 10 deg or > 10 mm. Poses in between ("borderline") are plotted but
do not set the threshold.

Usage (from the project root):
  python code/vgap/recommend_tau_c.py data/calib/calib_*.csv
  python code/vgap/recommend_tau_c.py --sim                  # starting values
  python code/vgap/recommend_tau_c.py data/calib/*.csv --no-injected
Writes a text report and one histogram PNG per object next to --out.
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OBJECTS = ["gearbox", "l_block", "t_block", "hemisphere"]
GOOD_ROT, GOOD_TRANS = 5.0, 5.0
BAD_ROT, BAD_TRANS = 10.0, 10.0
EPS = 0.005                 # tau_c = max wrong C + EPS, rounded up to 0.005
OTHER_CHECK_REASONS = {"ambiguous orientation", "ambiguous class",
                       "axis ambiguous"}
CURRENT_TAU = dict(gearbox=0.80, l_block=0.85, t_block=0.85, hemisphere=0.85)


def _f(x):
  try:
    v = float(x)
    return v if v == v else None
  except (TypeError, ValueError):
    return None


def label_from_error(obj, final_obj, rot, trans):
  if final_obj and final_obj != obj:
    return "wrong"
  if rot is None or trans is None:
    return "unsure"
  if rot <= GOOD_ROT and trans <= GOOD_TRANS:
    return "correct"
  if rot > BAD_ROT or trans > BAD_TRANS:
    return "wrong"
  return "borderline"


# ------------------------------------------------------------------ loading
def load_calib_csv(paths, use_injected=True, use_raw=False):
  rows, raw = [], []
  for p in paths:
    with open(p, newline="") as f:
      for r in csv.DictReader(f):
        inj = r.get("sample") == "injected"
        if inj and not use_injected:
          continue
        C = _f(r.get("C"))
        if C is None or r.get("label") not in ("correct", "wrong", "borderline"):
          continue
        d = dict(object=r["object"], color=r.get("color", ""),
                 tilt=r.get("tilt_deg", ""), C=C, label=r["label"],
                 IB=_f(r.get("IB")), IM=_f(r.get("IM")),
                 sample=r.get("sample", "real"),
                 perturbation=r.get("perturbation", ""),
                 caught=r.get("caught_by_other_check") == "True",
                 source=os.path.basename(p))
        if inj and r.get("inject_kind") == "raw":
          d["group"] = r.get("inject_group", "")
          if r["label"] == "wrong":
            raw.append(d)
          if not (use_raw and r.get("inject_group") == "in_plane"):
            continue
        rows.append(d)
  return rows, raw


def load_sim(use_hemi_fix=True):
  """Every (C, pose error) pair the simulations produced with the current
  pipeline: main path and, where it ran, the fallback answer."""
  rows = []
  stress = os.path.join(HERE, "sim_stress_results", "trials.csv")
  hemi = os.path.join(HERE, "sim_stress_results", "hemisphere_axis_fix_trials.csv")

  def add(r, tag, src):
    C = _f(r.get(f"C_{tag}"))
    if C is None or r.get(f"state_{tag}") in (None, "", "skip"):
      return
    lab = label_from_error(r["object"], r.get(f"class_{tag}"),
                           _f(r.get(f"rot_err_{tag}")), _f(r.get(f"trans_err_{tag}")))
    if lab == "unsure":
      return
    rows.append(dict(object=r["object"], color=r["color"], tilt=r["tilt"], C=C,
                     label=lab, sample="sim",
                     caught=r.get(f"reason_{tag}", "") in OTHER_CHECK_REASONS,
                     source=src))

  for path, keep in ((stress, lambda o: o != "hemisphere" or not use_hemi_fix),
                     (hemi, lambda o: o == "hemisphere")):
    if not os.path.isfile(path):
      continue
    with open(path, newline="") as f:
      for r in csv.DictReader(f):
        if not keep(r["object"]):
          continue
        add(r, "main", os.path.basename(path))
        if r.get("used_fallback") == "True":
          add(r, "fb", os.path.basename(path))
  return rows


# ------------------------------------------------------------------ analysis
def recommend(rows):
  """Returns dict with the zero-false-accept tau_c and its true-accept rate."""
  good = sorted(r["C"] for r in rows if r["label"] == "correct")
  wrong_all = [r["C"] for r in rows if r["label"] == "wrong"]
  wrong = [r["C"] for r in rows if r["label"] == "wrong" and not r["caught"]]
  out = dict(n_correct=len(good), n_wrong=len(wrong_all),
             n_wrong_uncaught=len(wrong),
             n_border=sum(r["label"] == "borderline" for r in rows),
             min_correct=good[0] if good else None,
             max_wrong=max(wrong) if wrong else None,
             max_wrong_all=max(wrong_all) if wrong_all else None)
  if not wrong:
    out["tau"] = None
    return out
  tau = math.ceil((max(wrong) + EPS) / 0.005) * 0.005
  out["tau"] = round(min(tau, 1.0), 3)
  out["tar"] = sum(c >= out["tau"] for c in good) / len(good) if good else None
  out["gap"] = (good[0] - max(wrong)) if good else None   # > 0: fully separable
  return out


def rate_at(rows, tau):
  good = [r["C"] for r in rows if r["label"] == "correct"]
  wrong = [r["C"] for r in rows if r["label"] == "wrong" and not r["caught"]]
  tar = sum(c >= tau for c in good) / len(good) if good else float("nan")
  return tar, sum(c >= tau for c in wrong)


def weight_sweep(by_obj, step=0.05):
  """C = wB*IB + wM*IM with wB = 1 - wM. For every wM, the zero-false-accept
  tau_c and its true-accept rate; best = most correct poses accepted, then
  the widest gap. Only rows that logged IB and IM take part.
  Approximate: the pipeline also uses C to pick between pose hypotheses and
  classes, and a different wM could change those picks."""
  ws = [round(i * step, 2) for i in range(int(round(1 / step)) + 1)]
  L = ["Best IoU weights (C = wB*IB + wM*IM, wB = 1 - wM; zero false accepts, "
       "then most true accepts, then widest gap):",
       f"{'object':11s} {'best wM':>7s} {'tau_c':>6s} {'accept':>7s} {'gap':>7s}"
       f"   | now wM 0.70: {'tau_c':>6s} {'accept':>7s} {'gap':>7s}"]
  per_w = defaultdict(list)
  for obj in [o for o in OBJECTS if o in by_obj]:
    rs = [r for r in by_obj[obj] if r.get("IB") is not None and r.get("IM") is not None]
    if not rs:
      continue
    res = {}
    for w in ws:
      rr = [dict(r, C=(1 - w) * r["IB"] + w * r["IM"]) for r in rs]
      rec = recommend(rr)
      res[w] = rec
      per_w[w].append(rec.get("tar") or 0.0)
    ok = [w for w in ws if res[w].get("tau") is not None]
    if not ok:
      L.append(f"{obj:11s} no wrong poses to separate from")
      continue
    best = max(ok, key=lambda w: (round(res[w]["tar"], 3), res[w]["gap"]))
    b, n = res[best], res.get(0.7, {})
    L.append(f"{obj:11s} {best:7.2f} {fmt(b['tau']):>6s} {100*b['tar']:6.0f}% "
             f"{fmt(b['gap']):>7s}   |             {fmt(n.get('tau')):>6s} "
             f"{(100*n['tar'] if n.get('tar') is not None else float('nan')):6.0f}% "
             f"{fmt(n.get('gap')):>7s}")
  if per_w:
    wbest = max(per_w, key=lambda w: (np.mean(per_w[w]), -abs(w - 0.7)))
    L.append(f"One shared wM for all objects: {wbest:.2f} (mean accept "
             f"{100*np.mean(per_w[wbest]):.0f}% vs {100*np.mean(per_w[0.7]):.0f}% at 0.70)."
             " Then use each object's tau_c recomputed at that wM (rerun with it).")
  return "\n".join(L)


def plot(rows, obj, rec, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  bins = [i / 100 for i in range(30, 101, 1)]
  groups = [("correct", "#2a9d8f", lambda r: r["label"] == "correct"),
            ("wrong (no other check catches it)", "#e63946",
             lambda r: r["label"] == "wrong" and not r["caught"]),
            ("wrong (caught by margin/class/axis check)", "#f4a261",
             lambda r: r["label"] == "wrong" and r["caught"]),
            ("borderline 5-10 deg / mm", "#8d99ae",
             lambda r: r["label"] == "borderline")]
  fig, ax = plt.subplots(figsize=(7, 3.6))
  for name, col, sel in groups:
    vals = [max(0.3, r["C"]) for r in rows if sel(r)]
    if vals:
      ax.hist(vals, bins=bins, alpha=0.65, color=col, label=f"{name} (n={len(vals)})")
  if rec.get("tau") is not None:
    ax.axvline(rec["tau"], color="k", lw=1.5, label=f"recommended tau_c = {rec['tau']:.3f}")
  if obj in CURRENT_TAU:
    ax.axvline(CURRENT_TAU[obj], color="k", ls=":", lw=1,
               label=f"current tau_c = {CURRENT_TAU[obj]:.2f}")
  ax.set_xlabel("C = 0.3 IB + 0.7 IM  (C < 0.3 drawn at 0.3)")
  ax.set_ylabel("poses")
  ax.set_title(obj)
  ax.legend(fontsize=7, loc="upper left")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def fmt(x, nd=3):
  return "-" if x is None else f"{x:.{nd}f}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("csv", nargs="*", help="calibration CSVs (globs ok)")
  ap.add_argument("--sim", action="store_true", help="use the simulation results")
  ap.add_argument("--no-injected", action="store_true",
                  help="ignore injected-error rows; real pipeline outputs only")
  ap.add_argument("--use-raw", action="store_true",
                  help="also let raw in-plane injected errors set tau_c")
  ap.add_argument("--sweep-weights", action="store_true",
                  help="also search wM (wB = 1 - wM) for the best separation")
  ap.add_argument("--out", default=None, help="output folder")
  a = ap.parse_args()

  paths = sorted({p for g in a.csv for p in glob.glob(g)})
  rows, raw = load_calib_csv(paths, not a.no_injected, a.use_raw) if paths \
      else ([], [])
  if a.sim:
    rows += load_sim()
  if not rows:
    raise SystemExit("No labelled rows. Pass calibration CSVs or --sim.")
  out = a.out or os.path.join(HERE, "tau_c_sim" if a.sim and not paths else "tau_c_calib")
  os.makedirs(out, exist_ok=True)

  by_obj = defaultdict(list)
  for r in rows:
    by_obj[r["object"]].append(r)
  # An object without ICP (hemisphere) never produces "after ICP" wrong
  # poses; its raw in-plane errors (e.g. a shifted sphere fit) stand in.
  stand_in = []
  for obj in list(by_obj):
    if not any(r["label"] == "wrong" and not r["caught"] for r in by_obj[obj]):
      extra = [r for r in raw if r["object"] == obj and r["group"] == "in_plane"]
      if extra:
        by_obj[obj] += extra
        stand_in.append(obj)
  src = "SIMULATION (starting values; replace with real data)" if a.sim and not paths \
      else ", ".join(os.path.basename(p) for p in paths) + (" + sim" if a.sim else "")
  L = [f"tau_c recommendation. Data: {src}",
       f"Correct <= {GOOD_ROT:g} deg & <= {GOOD_TRANS:g} mm; wrong = wrong class or "
       f"> {BAD_ROT:g} deg / > {BAD_TRANS:g} mm.",
       "tau_c = lowest threshold with zero false accepts among wrong poses no other "
       "check catches.", ""]
  hdr = (f"{'object':11s} {'tau_c':>6s} {'accept':>7s} {'n_ok':>5s} {'n_bad':>6s} "
         f"{'min C ok':>9s} {'max C bad':>10s} {'gap':>7s}   now: tau  accept  FA")
  L += [hdr, "-" * len(hdr)]
  per_color = []
  for obj in [o for o in OBJECTS if o in by_obj] + \
             sorted(set(by_obj) - set(OBJECTS)):
    rs = by_obj[obj]
    rec = recommend(rs)
    now = CURRENT_TAU.get(obj)
    tar_now, fa_now = rate_at(rs, now) if now else (float("nan"), 0)
    L.append(f"{obj:11s} {fmt(rec['tau']):>6s} "
             f"{(fmt(100*rec['tar'],0)+'%') if rec.get('tar') is not None else '-':>7s} "
             f"{rec['n_correct']:5d} {rec['n_wrong_uncaught']:3d}/{rec['n_wrong']:<3d}"
             f"{fmt(rec['min_correct']):>9s} {fmt(rec['max_wrong']):>10s} "
             f"{fmt(rec.get('gap')):>7s}   {fmt(now,2):>8s} {100*tar_now:5.0f}% {fa_now:3d}")
    if rec["tau"] is None:
      L.append(f"  ! {obj}: no wrong poses that only C can catch -> tau_c is not "
               "pinned down; collect wrong examples (injected errors, hard views).")
    elif rec["n_wrong_uncaught"] < 30:
      L.append(f"  ! {obj}: only {rec['n_wrong_uncaught']} wrong poses set this; with "
               "zero false accepts in n trials the true rate can still be up to ~3/n.")
    plot(rs, obj, rec, os.path.join(out, f"C_hist_{obj}.png"))
    for col in sorted({r["color"] for r in rs}):
      rc = recommend([r for r in rs if r["color"] == col])
      per_color.append(f"  {obj:11s} {col:10s} tau {fmt(rc['tau'])}  "
                       f"n_ok {rc['n_correct']:3d}  n_bad {rc['n_wrong_uncaught']:3d}  "
                       f"min C ok {fmt(rc['min_correct'])}  max C bad {fmt(rc['max_wrong'])}")
  for obj in stand_in:
    L.append(f"  * {obj}: no ICP, so its raw in-plane injected errors are used as "
             "the wrong poses.")
  L += ["", "n_bad = wrong poses only C can catch / all wrong poses. gap > 0 means "
        "every correct pose scores above every such wrong pose.",
        "accept = share of correct poses accepted at that tau_c.", "",
        "Per color (use the per-object value; this shows whether one color drives it):"]
  L += per_color
  if raw:
    L += ["", "Sensitivity: share of raw injected errors REJECTED (C < tau_c) at the "
          "recommended / current tau_c", "(pose deliberately moved off the true one, "
          "no ICP; 'tilt' and 'shift z' are out of the image plane)"]
    recs = {o: recommend(by_obj[o]).get("tau") for o in by_obj}
    for obj in [o for o in OBJECTS if o in by_obj]:
      pert = sorted({r["perturbation"] for r in raw if r["object"] == obj})
      cells = []
      for pn in pert:
        cs = [r["C"] for r in raw if r["object"] == obj and r["perturbation"] == pn]
        t, now = recs.get(obj), CURRENT_TAU.get(obj, 0.85)
        rec_s = f"{100*sum(c < t for c in cs)/len(cs):.0f}%" if t else "-"
        cells.append(f"{pn.replace(' raw', ''):>13s} {rec_s:>4s}/"
                     f"{100*sum(c < now for c in cs)/len(cs):.0f}%")
      L.append(f"  {obj:11s}" + "".join(
          ("\n" + " " * 13 if i and i % 4 == 0 else "") + c
          for i, c in enumerate(cells)))
  if a.sweep_weights:
    L += ["", weight_sweep(by_obj)]
  rep = "\n".join(L)
  with open(os.path.join(out, "tau_c_report.txt"), "w") as f:
    f.write(rep + "\n")
  print(rep)
  print(f"\n[+] Report and histograms in {out}")


if __name__ == "__main__":
  main()
