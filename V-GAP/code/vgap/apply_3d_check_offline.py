"""Applies the 3D check to recorded gate runs (offline) and compares it with
the gate as run (C, margin, class and axis checks only).

F3 = share of scan points within 3 mm of the posed CAD; it is the
within_3mm column every burst already logged, the same number
vgap_pipeline.scan_fit_3d() computes live. A burst that was Valid becomes
Conflict "3D check" when F3 < f3_min (0.7; hemisphere 0.6).
Limitation: a live 3D rejection would also trigger the table yaw-sweep
fallback, which might find another pose; offline the burst just counts as
rejected, so the accept counts here are a lower bound.

Usage (from data/gate_eval_all):  python ../../code/vgap/apply_3d_check_offline.py
Writes 3d_check_offline.txt and figures/fig6_3d_check_applied.png.
"""
import csv, glob, collections as c
from math import sqrt

F3_MIN = {"gearbox": 0.7, "l_block": 0.7, "t_block": 0.7, "hemisphere": 0.6}
OBJ = [("gearbox", "Gearbox"), ("hemisphere", "Hemisphere"), ("l_block", "L block"), ("t_block", "T block")]
OUT = {("correct", True): "TA", ("wrong", True): "FA", ("borderline", True): "AO",
       ("correct", False): "FR", ("wrong", False): "TR", ("borderline", False): "RO"}


def wilson(k, n, z=1.96):
  if n == 0:
    return (0.0, 0.0)
  p = k / n; d = 1 + z * z / n
  cen = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return max(0.0, cen - h), min(1.0, cen + h)


rows = []
for f in sorted(glob.glob("*/gate_eval.csv")):
  rows += list(csv.DictReader(open(f)))

def outcomes(rs, with3d):
  cnt = c.Counter()
  for r in rs:
    if r["state"] == "no detection":
      cnt["no detection"] += 1; continue
    ok = r["state"] == "Valid"
    if with3d and ok:
      try:
        ok = float(r["within_3mm"]) >= F3_MIN[r["object"]]
      except ValueError:
        ok = False
    cnt[OUT[(r["grade"], ok)]] += 1
  return cnt

L = ["3D check applied offline to the recorded gate runs",
     f"bursts: {len(rows)}   rule: Valid only if C >= tau_c, margin >= delta, other checks pass AND "
     "F3 >= 0.7 (hemisphere 0.6); F3 = share of scan points within 3 mm of the posed CAD", ""]
def line(name, rs):
  out = []
  for tag, w in (("gate as run", False), ("+ 3D check", True)):
    k = outcomes(rs, w); acc = k["TA"] + k["AO"] + k["FA"]; good = k["TA"] + k["FR"]
    lo, hi = wilson(k["FA"], acc)
    out.append(f"  {name:22s} {tag:12s} accepted {acc:5d}  correct {k['TA']:5d}  a bit off {k['AO']:3d}  "
               f"WRONG {k['FA']:2d} ({100*k['FA']/max(acc,1):.2f}%, 95% CI {100*lo:.2f}-{100*hi:.2f}%)  "
               f"good poses rejected {k['FR']:3d}/{good} ({100*k['FR']/max(good,1):.1f}%)")
  return out
L += line("ALL", rows)
L.append("")
for k, n in OBJ:
  L += line(n, [r for r in rows if r["object"] == k])
L.append("")
for k, n in OBJ:
  for col in ("white", "grey", "reflective"):
    rs = [r for r in rows if r["object"] == k and r["color"] == col]
    if rs:
      L += line(f"{n} / {col}", rs)
L.append("")
ch = [r for r in rows if r["state"] == "Valid" and r["within_3mm"] and
      float(r["within_3mm"]) < F3_MIN[r["object"]]]
L.append(f"Accepted poses the 3D check would reject: {len(ch)}")
for r in ch:
  L.append(f"  {r['object']} {r['color']} {r['view']} rep {r['rep']}: grade {r['grade']}, C {float(r['C']):.3f}, F3 {float(r['within_3mm']):.3f}")
corr = [float(r["within_3mm"]) for r in rows if r["state"] == "Valid" and r["grade"] == "correct" and r["within_3mm"]]
corr.sort()
L.append(f"F3 on accepted correct poses: n {len(corr)}, min {corr[0]:.3f}, 1st percentile {corr[len(corr)//100]:.3f}, "
         f"median {corr[len(corr)//2]:.3f}")
txt = "\n".join(L); open("3d_check_offline.txt", "w").write(txt + "\n"); print(txt)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
  if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555", "xtick.color": "#555", "ytick.color": "#555"})
fig, ax = plt.subplots(figsize=(6, 3.15), dpi=200)
x = range(len(OBJ)); w = 0.36
for dx, with3d, col, lab in ((-w/2, False, "#EB6834", "Gate as run (no 3D check)"), (w/2, True, "#2A79D6", "Gate + 3D check")):
  vals = [outcomes([r for r in rows if r["object"] == k], with3d)["FA"] for k, _ in OBJ]
  b = ax.bar([i + dx for i in x], vals, w, color=col, label=lab, zorder=2); ax.bar_label(b, padding=2, fontsize=9)
ax.set_xticks(list(x), [n for _, n in OBJ]); ax.set_ylim(0, 5); ax.set_ylabel("Wrong poses accepted")
ax.grid(axis="y", color="#e3e3e3", zorder=0)
ax.set_title("Wrong poses accepted, 1,440 recorded bursts", loc="left", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig6_3d_check_applied.png")
