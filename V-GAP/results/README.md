# Results

Every experiment on the UR5e, one folder per set. Each run folder is one object/color run of `code/vgap/vgap_gate_eval_ur5e.py` (grasp trials: `code/vgap/vgap_grasp_trials_ur5e.py`).

| Folder | Experiment | Headline |
|---|---|---|
| `main_matrix_0_30_45/` | 4 objects x 3 colors x 0/30/45 degree views x flat/propped, 1,440 bursts | 3 wrong of 1,287 accepted (0 with the 3D check); plain RANSAC+ICP 644 wrong. Summary: `RESULTS.md`, figures in `figures/` |
| `static_60deg/` | Arm fixed at 60 degrees, 480 bursts | 0 wrong of 424 accepted (409 correct, 15 slightly off). Use `combined_gate_report.txt` (per-run reports were removed because they predate the regrade) |
| `occlusion_recovery/` | White parts at 45 degrees behind a board (A) or a board + bottle (B), 5 reps | A 20/20 and B 20/20 recovered, 0 wrong |
| `hemisphere_axis15/` | Propped hemisphere with axis-ambiguity limit 15 degrees | 30/30 accepted correct |
| `gocator_ground_truth/` | Every pose compared with an independent CAD fit on a Gocator 3210 scan (white, 0/30/45 + 60 static; reflective T extra) | 160 white bursts, 159 accepted, 0 wrong. `not_used/` holds runs whose Gocator reference was invalid (part outside the scanner's view) |
| `grasp_trials/` | 80 paired grasps, V-GAP vs plain RANSAC+ICP, Robotiq 2F-140 | 39/40 vs 15/40. Summary: `GRASP_RESULTS.md` |
| `depth_before_after/` | Old vs new RealSense laser/exposure on the white gearbox | Old settings: every burst rejected; new: valid |

## Files in a run folder

| File | Contents |
|---|---|
| `gate_eval.csv` | One row per burst: view, YOLO class and confidence, gate state and reason, C / IB / IM, 3D fit (`within_3mm` = F3), pose in camera and base frame, operator grade, outcome, the plain RANSAC+ICP pose and which step changed it |
| `injected.csv` | Deliberately wrong poses (rotations, shifts, tilts) pushed through the gate at each view, with the gate decision (not in runs made with `--no-inject`) |
| `recovery.csv` | Recovery attempts after a Conflict: stage, from/to angle, old and new gate state, C and grade (only in runs with recovery) |
| `gocator.csv` | Gocator reference pose and its validity checks (fit, repeatability); `gocator_as_run.csv` is the reference before a documented regrade |
| `gate_report.txt`, `gate_summary.csv` | Report built by `--report` |
| `combined_gate_report.txt` | Report over all runs in a set (`--report <folder>/*/`) |

Outcome codes: **TA** accepted and correct, **FA** accepted but wrong, **AO** accepted but a bit off (5-10 mm or degrees), **TR** correctly rejected (wrong pose), **FR** rejected although correct, **RO** rejected and a bit off. Grading: correct within 5 mm / 5 degrees of the reference pose, wrong beyond 10 mm / 10 degrees or wrong shape.

Reference poses: operator-confirmed poses for the main sets; Gocator CAD fits for `gocator_ground_truth/`. Raw images and point clouds are not included.
