# Gocator 3210 ground truth for V-GAP

> Development log kept during the Gocator work (Nov 2026). Backup copies and `*_before_regrade.csv` files it mentions are not in this repo. In `results/gocator_ground_truth/` each run folder holds the final (regraded) files, and `gocator_as_run.csv` is the reference before the regrade. Validity rule: a Gocator reference counts only if at least 95% of its points are within 1 mm of the CAD fit and repeat scans agree within 2 degrees / 2 mm.

## Files to copy to the Jetson (same folder as Gocator.py and GoSdk_MsgHandler.py)
- `code/vgap/gocator_cam.py` – one Gocator snapshot -> points (m). Uses your SDK setup (GO_SDK_4 path from gocator_scan.py, IP 192.168.1.10, Uniform Surface output).
- `code/vgap/gocator_handeye.py` – Gocator hand-eye (T_flange_goc), arm is only READ, never commanded.
- `code/vgap/gocator_gt.py` – reference pose from a Gocator scan, independent of V-GAP.
- `code/vgap_gate_eval_ur5e.py` – new `--gocator` mode + report section L.
Needs numpy, open3d, scipy, opencv (all already used by the pipeline except maybe scipy: `pip install scipy`).

## How the reference is made (so it can't copy a bad V-GAP pose)
1. Table plane removed, largest cluster kept.
2. CAD fitted from ~600 start orientations + FPFH RANSAC (scan -> CAD ICP, 1 mm). V-GAP's pose is NOT used.
3. Best fit = reference. `gt_margin` = how much better it fits than the next different pose (big = unambiguous).
4. Hemisphere: dome is a sphere, so its surface can't fix the axis. Center = sphere fit (exact); axis = table normal, valid only when flat. Propped hemisphere is graded on position only.
5. Each view scanned 3x: `gt_repeat_mm/deg` = how much the reference moves between scans.
Sim check (synthetic Gocator scans, 0/30/45 deg, flat + propped): gearbox, L, T found to 0.0-0.4 deg / <0.01 mm; hemisphere center <0.01 mm, axis exact when flat. Full --gocator mode test: Gocator-graded errors matched the simulator's true errors within 0.3 deg / 0.03 mm.

## Steps
**0. Capture test** (Gocator glass ~20-22 cm above the TABLE, so table and object are both inside the 164-274 mm range; if only the object top is in range it gets removed as "the table"):
```
python gocator_cam.py --n 3
python gocator_gt.py --object l_block --scan gocator_test.ply
```
Expect "within 1 mm" ~0.95-1.0 and RMSE ~0.4-0.5 mm (limited by the 1 mm CAD point spacing).

**1. Gocator hand-eye** (hemisphere print flat on the table, don't move it):
```
python gocator_handeye.py --robot-ip 192.168.1.5
```
Jog (freedrive) to 10-12 poses, Enter at each. Keep the glass 22-27 cm from the TABLE in every pose (dome is 50 mm tall). Vary the tilt (0-35 deg, different sides) AND where the hemisphere is in the view (edges, near ~22 cm, far ~27 cm). If it's always centered at the same distance, the rotation can't be solved even though the residual looks fine (the script warns). Good: mean residual < ~0.5 mm. Saves `data/handeye/T_flange_goc.npy` + `.json`.

**2. Gate eval with the Gocator reference** (a subset is enough; it validates your grading for the full 1,440):
```
G="--robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam_updated_zfix.npy --gocator-hand-eye data/handeye/T_flange_goc.npy --tool-tip 0,0,0.086 --table-z 0.01 --angles 0,30,45 --placements flat,propped --orientations 1 --reps 5 --gocator --no-target-filter"
python vgap_gate_eval_ur5e.py --object l_block --color white $G --out data/gate_eval_gocator/l_block_white
python vgap_gate_eval_ur5e.py --object t_block --color white $G --out data/gate_eval_gocator/t_block_white
python vgap_gate_eval_ur5e.py --object gearbox --color white $G --out data/gate_eval_gocator/gearbox_white
python vgap_gate_eval_ur5e.py --object hemisphere --color white $G --axis-ambig-deg 15 --out data/gate_eval_gocator/hemisphere_white
python vgap_gate_eval_ur5e.py --report data/gate_eval_gocator/*/ --out data/gate_eval_gocator
```
- Use the RealSense hand-eye you actually trust (the ArUco one above; change the path if yours differs).
- `--table-z 0.01` = table height in the base frame (measured 11/01: the Gocator hand-eye puts the flat hemisphere center, i.e. the table, at z 0.008 m; the ArUco hand-eye put the marker at 13.8 mm). It makes the clearance checks real now that the arm goes close. Keep Enter-to-move on (no `--no-confirm`) and watch the first Gocator move.
- Frame note (11/01): the Gocator's output origin is NOT at the sensor (it sits near the middle of the range, Z toward the sensor). The gate eval now places each object at the AVERAGE spot where the hand-eye calibration saw the sphere (`center_gocator_m` in T_flange_goc.json), shifted `--gocator-dz` m along the tool axis (+ = farther away). Old `--gocator-standoff` only applies to an old calibration json.
- Gearbox is 65 mm tall: with the default `--gocator-dz 0` its center sits at the calibration spot (~230 mm from the glass, measured 11/01), so the top is ~198 mm and the table ~262 mm out, both inside 164-274 mm. Do NOT add +0.01 (table would reach ~272 mm, the edge of the range). It is 153 mm long, so the scan sees only part of it (FOV ~90x130 mm); that is fine for the fit.
- If a view prints "[gocator] view unsafe ...", lower `--gocator-dz` (closer) only if the tip check is the problem and you have checked it is safe, or lower `--min-tip-object`.
- Add grey / reflective if time allows (same commands, `--color`).

## What you get
- `gate_eval.csv`: per burst `gt_rot_err_deg`, `gt_trans_err_mm`, `gt_grade`, `gt_outcome`, reference quality (`gt_fitness_1mm`, `gt_rmse_mm`, `gt_margin`, `gt_repeat_*`), and the plain RANSAC+ICP pose graded the same way (`baseline_gt_*`).
- `gocator.csv`: one row per view (reference pose in the base frame, quality). Raw scans in `gocator/*.npy` (can be re-analysed offline).
- Report section L: errors of accepted poses vs the Gocator, gate outcomes graded by the Gocator (FA rate, precision, FR rate), agreement between your grades and the Gocator grades, plain vs V-GAP wrong poses.

## Caveat for the paper
The error includes both hand-eye calibrations (RealSense ArUco ~1.5 mm, Gocator from step 1). Report them next to the errors. A constant offset across all views of an object is a calibration bias, not a V-GAP error.

## Static 60 deg with the Gocator (added 11/01)
`--static --gocator` now works: the arm never moves; after the RealSense bursts at each position the Gocator scans from the SAME pose. Put the arm by hand so the object is ~22-27 cm from the Gocator glass (check with `python gocator_cam.py`: the z span should include table + object) AND the RealSense still sees it. V-GAP and the Gocator share the same TCP pose, so `--static-tcp` is not needed for the comparison. Sim check: Gocator-graded error matched the true error within 0.4 deg / 0.01 mm.
```
IMPORTANT (11/02): pass the arm's real pose with --static-tcp. The Gocator reference finds the table from the arm pose; without it the pose is identity and the reference fails. Read it (prints x,y,z,rx,ry,rz in m/rad) and re-read whenever the arm is moved:
python -c "import rtde_receive as r; p=r.RTDEReceiveInterface('192.168.1.5').getActualTCPPose(); print(','.join(f'{v:.6f}' for v in p))"
TCP="<paste>"
S="--static --static-tcp $TCP --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam_updated_zfix.npy --gocator --gocator-hand-eye data/handeye/T_flange_goc.npy --placements flat,propped --orientations 1 --reps 5 --conditions normal"
python vgap_gate_eval_ur5e.py --object l_block --color white $S --out data/gate_eval_gocator_static60/l_block_white
python vgap_gate_eval_ur5e.py --object t_block --color white $S --out data/gate_eval_gocator_static60/t_block_white
python vgap_gate_eval_ur5e.py --object gearbox --color white $S --out data/gate_eval_gocator_static60/gearbox_white
python vgap_gate_eval_ur5e.py --object hemisphere --color white $S --axis-ambig-deg 15 --out data/gate_eval_gocator_static60/hemisphere_white
python vgap_gate_eval_ur5e.py --report data/gate_eval_gocator_static60/*/ --out data/gate_eval_gocator_static60
```

## 11/02 fix: big object top taken as the table (T at 30/45 deg)
- At tilted views the T's top face filled more of the 90x150 mm frame than the visible table, so the largest plane (the "table") was the T top and the table was fitted as the object (fitness 0.41-0.83). The T itself was in frame.
- Fix (gocator_gt.isolate_object): normal oriented toward the sensor (view_dir = calibrated tool axis, +z by default), then if many points lie BEYOND that plane and form a parallel plane, that farther plane is the table. a30 flat scan: fitness 0.413 -> 0.998, margin 0.37. All sims pass.
- Guard: references with fitness < 0.95 (--gt-min-fitness) or repeat scans > 2 deg / 2 mm are marked invalid and not graded (in-run and in --report).
- gocator.csv now logs gt_tcp (Gocator arm pose).
- Regrade a finished run from its saved scans (no robot), from the folder you run the gate eval in:
  python code/vgap/gocator_regrade.py --run data/gate_eval_gocator/t_block_white --hand-eye matrix_config/T_flange_cam_updated.npy --gocator-hand-eye data/handeye/T_flange_goc.npy
  (OLD runs (L/T 11/01): pass the OLD camera matrix they were run with, not _zfix. Rebuilds the Gocator pose from tcp_pose + hand-eyes + --dist 0.40 for old runs; sim check: rebuilt reference = original within 0.00 mm / 0.05 deg; keeps *_before_regrade.csv)

## 11/02 fix 2: gearbox (all 6 views failed)
- Causes: (1) the pipeline CAD cloud (60k samples, exterior only -> 25k) has ~2.3 mm gaps on the big gearbox, so even a perfect fit scored ~0.78 "within 1 mm" (sim reproduced 0.78); (2) the gearbox fills the 90x150 mm frame, so little/no table is visible ("almost nothing above the largest flat surface").
- Fixes: CadTarget(cad, cfg) resamples the mesh at 400k points (sim gearbox 0.78 -> 0.999); table plane now comes from the arm pose + --gocator-table-z (default 0.008 m, Gocator hand-eye) and is refined on the table points it sees (--gocator-find-table = old search). Real T a30 flat: same reference as before (0.27 mm), fitness 1.000. Sims gearbox/T/hemisphere pass.
- Regrade gearbox (run used the _zfix camera matrix): python code/vgap/gocator_regrade.py --run data/gate_eval_gocator/gearbox_white --hand-eye matrix_config/T_flange_cam_updated_zfix.npy --gocator-hand-eye data/handeye/T_flange_goc.npy

## 11/02 fix 3: T cut off at the field-of-view edge
T a30 propped and a45 flat references stayed at fitness ~0.72. Cause: the T crossbar and stem ran out of the Gocator frame (scan edges at the sensor FOV), so the scan centroid was ~15-30 mm from the CAD centroid and every centroid-aligned start converged to a wrong pose. gocator_gt.gt_pose now adds a fallback when the best coarse fit < 0.95: CAD flat face on the table normal (both ways up), yaw every 10 deg, in-plane offsets up to +-30 mm (3,528 starts, ~25 s). Both views now fit 1.000 (RMSE 0.38 mm). V-GAP vs reference at those views: rot 1.2-2.6 deg; trans 6.4-8.3 mm with the old camera matrix, 1.3-5.5 mm after the 6.3 mm z correction (9/10 within 5 mm). Lab tip: keep the object at the same table spot under the Gocator so it stays fully in frame.
