"""Recompute the Gocator reference of a finished --gocator gate run from its saved
scans (gocator/<view>_s<k>.npy) with the current gocator_gt.py, then regrade the
V-GAP poses against it and rerun the report. No robot or sensors needed.

Run it from the same folder you run vgap_gate_eval_ur5e.py from (CAD paths), e.g.
  python gocator_regrade.py --run data/gate_eval_gocator/t_block_white \\
      --hand-eye matrix_config/T_flange_cam_updated.npy \\
      --gocator-hand-eye matrix_config/T_flange_goc.npy

The Gocator's arm pose per view is read from gocator.csv (gt_tcp, logged since
11/02). Older runs did not log it, so it is rebuilt exactly the way the run
placed the Gocator: from the view's TCP (gate_eval.csv tcp_pose), the camera
hand-eye, --dist and the Gocator calibration spot. Pass the SAME --hand-eye,
--dist and --gocator-dz as the run. Views whose old reference was valid are
used as a check: the rebuilt reference must land on the old one (mm / deg).

Originals are kept as gate_eval_before_regrade.csv / gocator_before_regrade.csv.
The plain RANSAC+ICP Gocator grades (baseline_gt_*) cannot be redone (that pose
is not in the CSV), so they are cleared; the operator-graded comparison
(report section K) is unchanged.
"""
import argparse, csv, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_tau_c_ur5e as cal          # noqa: E402
import vgap_pipeline as vp                  # noqa: E402
import gocator_gt as ggt                    # noqa: E402
import vgap_gate_eval_ur5e as ge            # noqa: E402


def read(path):
  with open(path, newline="") as f:
    return list(csv.DictReader(f))


def write(path, rows):
  keys = list(dict.fromkeys(k for r in rows for k in r))
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)


def pose_from_fields(r, prefix):
  p = [float(r[f"{prefix}_{k}"]) / (1000 if k.endswith("mm") else 1)
       for k in ("x_mm", "y_mm", "z_mm", "rx", "ry", "rz")]
  return cal.from_ur_pose(p)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, help="folder with gate_eval.csv, gocator.csv, gocator/")
  ap.add_argument("--hand-eye", required=True, help="camera hand-eye used in the run")
  ap.add_argument("--gocator-hand-eye", required=True)
  ap.add_argument("--dist", type=float, default=0.40, help="same as the run (default 0.40)")
  ap.add_argument("--gocator-dz", type=float, default=0.0, help="same as the run")
  ap.add_argument("--static", action="store_true", help="run was --static (Gocator scanned from the view)")
  ap.add_argument("--gt-min-fitness", type=float, default=0.95)
  ap.add_argument("--gocator-table-z", type=float, default=0.008)
  ap.add_argument("--gocator-find-table", action="store_true")
  a = ap.parse_args()

  run = a.run.rstrip("/")
  ge_path, go_path = os.path.join(run, "gate_eval.csv"), os.path.join(run, "gocator.csv")
  for p in (ge_path, go_path):
    b = p.replace(".csv", "_before_regrade.csv")
    if not os.path.exists(b):
      shutil.copy(p, b)
  rows = read(ge_path.replace(".csv", "_before_regrade.csv"))
  grows = read(go_path.replace(".csv", "_before_regrade.csv"))

  X = np.load(a.hand_eye).astype(float)
  Xg = np.load(a.gocator_hand_eye).astype(float)
  with open(a.gocator_hand_eye.replace(".npy", ".json")) as f:
    info = json.load(f)
  g_axis = np.asarray(info["tool_axis_gocator"], float)
  g_spot = np.asarray(info["center_gocator_m"], float) + a.gocator_dz * g_axis

  obj = rows[0]["object"]
  # views whose Gocator step crashed during the run have scans but no gocator.csv row
  have = {g["view"] for g in grows}
  for v in dict.fromkeys(r["view"] for r in rows):
    if v not in have and any(f.startswith(v + "_s") for f in os.listdir(os.path.join(run, "gocator"))):
      grows.append(dict(object=obj, color=rows[0].get("color", ""), view=v))
  cad = vp.load_cad_model(vp.OBJECT_CONFIGS[obj])
  tgt = ggt.CadTarget(cad, vp.OBJECT_CONFIGS[obj])

  new_g = []
  for g in grows:
    view = g["view"]
    vrows = [r for r in rows if r["view"] == view]
    place = vrows[0]["placement"] if vrows else "flat"
    T_view = cal.from_ur_pose(json.loads(vrows[0]["tcp_pose"]))
    R = T_view[:3, :3]
    center = T_view[:3, 3] + R @ X[:3, 3] + a.dist * (R @ X[:3, :3])[:, 2]
    if g.get("gt_tcp"):
      T_g, src = cal.from_ur_pose(json.loads(g["gt_tcp"])), "logged"
    elif a.static:
      T_g, src = T_view, "static"
    else:
      T_g = np.eye(4)
      T_g[:3, :3] = R
      T_g[:3, 3] = center - R @ (Xg[:3, :3] @ g_spot + Xg[:3, 3])
      src = "rebuilt"
    T_s_b = np.linalg.inv(T_g @ Xg)
    near = None if a.static else T_s_b[:3, :3] @ center + T_s_b[:3, 3]

    scans = sorted(f for f in os.listdir(os.path.join(run, "gocator"))
                   if f.startswith(view + "_s") and f.endswith(".npy"))
    res = dict(gt_status="ok")
    try:
      if not scans:
        raise RuntimeError("no saved scans")
      Ts = []
      for k, fn in enumerate(scans):
        pts = np.load(os.path.join(run, "gocator", fn)).astype(float)
        tbl = None if a.gocator_find_table else ggt.table_in_sensor(T_g @ Xg, a.gocator_table_z)
        o, plane = ggt.isolate_object(pts, near=near, view_dir=g_axis, table=tbl)
        if k == 0:
          r0 = ggt.gt_pose(o, tgt, plane)
          T0 = r0["T_cad_to_sensor"]
          Ts.append(T0)
          res.update(gt_fitness_1mm=round(r0["fitness_1mm"], 4), gt_rmse_mm=round(r0["rmse_mm"], 3),
                     gt_margin=round(r0["margin"], 4) if r0["margin"] == r0["margin"] else "",
                     gt_n_points=r0["n_points"])
          rests = r0.get("rests_on_table", True)
        else:
          Ts.append(ggt.gt_refine(o, tgt, T0)[0])
      spr = [cal.pose_error(Ts[0], T, cad) for T in Ts[1:]]
      if spr:
        res.update(gt_repeat_deg=round(max(e[0] for e in spr), 3),
                   gt_repeat_mm=round(max(e[1] for e in spr), 3))
      res["gt_rot_valid"] = not (cad.axial_axis is not None and place != "flat")
      bad = []
      if not rests:
        bad.append("no fit rests on the table (part not fully seen)")
      if res["gt_fitness_1mm"] < a.gt_min_fitness:
        bad.append(f"fitness {res['gt_fitness_1mm']}")
      if spr and (res["gt_repeat_deg"] > 2 or res["gt_repeat_mm"] > 2):
        bad.append(f"repeat {res['gt_repeat_deg']} deg / {res['gt_repeat_mm']} mm")
      if bad:
        res["gt_status"] = "invalid: " + ", ".join(bad)
      T_ref = T_g @ Xg @ T0
    except Exception as e:
      res, T_ref = dict(gt_status=f"failed: {e}"), None

    old_ok = (g.get("gt_status") == "ok" and float(g.get("gt_fitness_1mm") or 0) >= a.gt_min_fitness
              and float(g.get("gt_repeat_deg") or 0) <= 2 and float(g.get("gt_repeat_mm") or 0) <= 2)
    chk = ""
    if old_ok and T_ref is not None:
      e_r, e_t = cal.pose_error(np.asarray(json.loads(g["gt_base"])), T_ref, cad)
      chk = f"  check vs old valid reference: {e_t:.2f} mm / {e_r:.2f} deg"
    print(f"[{view}] Gocator pose {src}; fitness {g.get('gt_fitness_1mm')} -> "
          f"{res.get('gt_fitness_1mm')}, {res['gt_status']}{chk}")

    ng = dict(g)
    ng.update({k: v for k, v in res.items()})
    ng["gt_tcp"] = json.dumps([round(v, 6) for v in cal.to_ur_pose(T_g)])
    ng["gt_base"] = json.dumps(np.round(T_ref, 6).tolist()) if T_ref is not None else ""
    new_g.append(ng)

    for r in vrows:
      for k in ge.GT_KEYS + ["gt_outcome"]:
        r[k] = ""
      r.update({k: v for k, v in res.items()})
      if T_ref is None or res["gt_status"] != "ok":
        continue
      if r.get("has_pose") not in ("True", True) or not r.get("base_x_mm"):
        r["gt_grade"] = "wrong"
      else:
        e_r, e_t = cal.pose_error(T_ref, pose_from_fields(r, "base"), cad)
        rot_ok = res["gt_rot_valid"]
        r["gt_grade"] = cal.label_of(e_r if rot_ok else 0.0, e_t,
                                     wrong_class=(r.get("final_object") or obj) != obj)
        r["gt_rot_err_deg"] = round(e_r, 2) if rot_ok else ""
        r["gt_trans_err_mm"] = round(e_t, 2)
      r["gt_outcome"] = ge.outcome(r["state"], r["gt_grade"])

  write(ge_path, rows)
  write(go_path, new_g)
  print("\n" + ge.report([run], os.path.join(run, "gate_report.txt"),
                         os.path.join(run, "gate_summary.csv")))
  print(f"\n[+] Regraded {run} (originals: *_before_regrade.csv)")


if __name__ == "__main__":
  main()
