"""Rebuilds fig2 and fig3 from the gate_eval.csv files (run from data/gate_eval_all).
2026-10-24: three T block poses regraded from 'a bit off' to wrong (depth check)."""
import csv, glob, collections as c
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
  if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555", "axes.labelcolor": "#111",
                     "xtick.color": "#555", "ytick.color": "#555"})
OR, BL = "#EB6834", "#2A79D6"
OBJ = [("gearbox", "Gearbox"), ("hemisphere", "Hemisphere"), ("l_block", "L block"), ("t_block", "T block")]
o, plain = c.defaultdict(c.Counter), c.Counter()
for f in glob.glob("*/gate_eval.csv"):
  for r in csv.DictReader(open(f)):
    o[r["object"]][r["outcome"]] += 1
    plain[r["object"]] += r["baseline_grade"] == "wrong"

fig, ax = plt.subplots(figsize=(6, 3.15), dpi=200)
x = range(len(OBJ)); w = 0.36
for dx, vals, col, lab in ((-w/2, [plain[k] for k, _ in OBJ], OR, "Plain RANSAC+ICP (no gate)"),
                           (w/2, [o[k]["FA"] for k, _ in OBJ], BL, "V-GAP after gate")):
  bars = ax.bar([i + dx for i in x], vals, w, color=col, label=lab, zorder=2)
  ax.bar_label(bars, padding=2, fontsize=9)
ax.set_xticks(list(x), [n for _, n in OBJ]); ax.set_ylim(0, 330); ax.set_ylabel("Wrong poses sent (of 360)")
ax.grid(axis="y", color="#e3e3e3", zorder=0); ax.set_title("Wrong poses that would reach the gripper", loc="left", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig2_wrong_poses_sent.png"); plt.close(fig)

cols = [("TA", "Accepted, correct (TA)", BL), ("AO", "Accepted, a bit off (AO)", "#EBA00C"),
        ("FA", "Accepted, WRONG (FA)", "#E04444"), ("FR", "Rejected, correct (FR)", OR),
        ("RO", "Rejected, a bit off (RO)", "#EB82A8"), ("TR", "Rejected, wrong (TR)", "#1AAE7A")]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
ys = list(range(len(OBJ)))[::-1]
for (k, _), y in zip(OBJ, ys):
  left = 0
  for key, lab, col in cols:
    v = o[k][key]
    ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=0.8, height=0.58,
            label=lab if y == ys[0] else None, zorder=2)
    left += v
ax.set_yticks(ys, [n for _, n in OBJ]); ax.set_xlim(0, 360); ax.set_xlabel("Bursts (360 per object)")
ax.grid(axis="x", color="#e3e3e3", zorder=0)
nfa = sum(o[k]["FA"] for k, _ in OBJ); nacc = sum(o[k][t] for k, _ in OBJ for t in ("TA", "AO", "FA"))
ax.set_title(f"Gate outcomes: 1,440 bursts, {nfa} false accepts ({nacc:,} accepted)", loc="left", fontsize=12)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig3_gate_outcomes.png"); plt.close(fig)
print("ok", nfa, nacc)
