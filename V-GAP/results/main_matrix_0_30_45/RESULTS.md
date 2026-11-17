# V-GAP gate evaluation: results (real UR5e, runs Oct 16-23, 2026)

Source: `combined_gate_report.txt` in this folder (14 runs, 1,440 bursts; 4 objects x 3 colors x 3 view angles x 2 placements x 4 orientations x 5 bursts). Per-run CSVs are in this folder; `gearbox_grey/` has only gate_eval.csv and injected.csv. tau_c: gearbox 0.90, L 0.85, T 0.81, hemisphere 0.85, wM 0.70.

## Headline
- 1,287 poses accepted: 1,268 correct, 16 "a bit off" (5-10 mm/deg), 3 wrong (0.23%, 95% CI 0.08-0.68%). All 3 wrong are T-block (2 reflective, 1 grey); regraded 2026-10-24 from "a bit off" after a depth check (see below).
- Plain RANSAC+ICP on the same captures: 644 of 1,440 wrong poses would have gone to the gripper (gearbox 81, L 290, T 273, hemisphere 0). V-GAP sent 3 (all T).
- Rescues, wrong plain pose -> correct V-GAP pose: table fallback 369, 90/180 deg hypothesis 265, L/T class check 42. Harmed: 1 (gearbox fallback).
- Accepted pose repeatability: 1-2 mm, 1-2 deg (gearbox 1.5 mm, L 1.1 mm, T 2.0 mm, hemisphere 0.1 mm). Scan points within 3 mm of CAD: 92-97%.

## Weak spots
- Hemisphere propped: 76 of 79 rejections were correct poses (FR 47% propped, 0% flat). All came from the axis check (axis ambiguity 10-15 deg vs limit 10). A limit of 15 deg removes all 76 and accepts only 3 extra "a bit off" poses, 0 wrong. That limit was chosen on this same data; a fresh propped-hemisphere set at 15 degrees (`../hemisphere_axis15/`) then gave 30/30 correct accepts.
- Reflective filament raises false rejects: gearbox 10%, L 12%, T 7.6% vs 0-6% for white/grey.
- C is blind to depth/tilt errors: of the injected wrong poses, C alone let 13-34% through, nearly all depth/tilt. Adding the 3D check (scan points within 3 mm >= 0.7, hemisphere 0.6) let 0 through (L: 1 of 929), with no extra false rejects, and it also rejects the 3 real false accepts.
- YOLO called the T an L 12% of the time; the class check fixed all 37 (98.3% right after).
- T-block false accepts (3): t_block_reflective_20261022_214410 a30_propped_o1 r5 and t_block_reflective_20261022_220932 a45_flat_o4 r3 (CAD turned ~90 deg about the stem; outline matches, depth off 3-5 cm), t_block_grey_20261022_211345 a45_propped_o4 r3 (~20 deg tilt). Only 51-67% of scan points within 3 mm (correct poses 92-97%); the 3D check at 0.7 rejects all 3. The hypothesis check only rotates about the height axis, so it cannot fix a turn about the stem. Overlay of one of these poses: `../../assets/t_pose_check_overlay.png`.
- Recovery was tested separately in the occlusion set (`../occlusion_recovery/`).

Figures: figures/fig1-fig5 (rebuild fig2/fig3 with `code/vgap/make_fig2_fig3.py`, fig4 with `code/vgap/make_fig4_injected.py`).

## With the 3D check applied (offline, 2026-10-24)
Script: `code/vgap/apply_3d_check_offline.py`; full table: 3d_check_offline.txt; figure: figures/fig6_3d_check_applied.png.
Rule: Valid only if the gate passed AND F3 >= 0.7 (hemisphere 0.6), F3 = share of scan points within 3 mm of the posed CAD (the logged within_3mm; same as vgap_pipeline.scan_fit_3d, now live with CHECK_3D = True).
- Accepted 1,287 -> 1,284; wrong 3 -> 0 (95% CI 0-0.30%); correct poses rejected unchanged (128 of 1,396, 9.2%). The only poses it removes are the 3 T false accepts (F3 0.51, 0.51, 0.67).
- Margin: lowest F3 on accepted correct poses: gearbox 0.90, L 0.89, T 0.85, hemisphere 0.65 (threshold 0.6, tight). Wrong T poses <= 0.67.
- Caveats: threshold chosen on this same data; a live rejection would also run the yaw-sweep fallback, which offline cannot be replayed (accept counts are a lower bound). The check runs live in all later sets (60 degree, occlusion, Gocator, grasp trials).
