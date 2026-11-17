# Occlusion / recovery test: commands (white, 4 objects, start at 45°)

Run from the folder that holds `vgap_gate_eval_ur5e.py` on the Jetson.
First copy the 3 new files there: `vgap_pipeline.py`, `vgap_gate_eval_ur5e.py`, `calibrate_tau_c_ur5e.py`.

Target filled in from the 10/26 gearbox capture (2nd position): x,y = 0.443,-0.559 (base frame, m). z = 0.146 for ALL objects (gearbox center height; lower views graze the board). The script keeps this z when it re-finds the center from above. Put every object on that same marked spot. The script also re-finds the center from straight above before the 45 deg view (if within 10 cm).
Put the board in place at the "occluded" prompt.
Put every object on the same marked spot so the target stays the same.

What the flags do here:
- `--angles 45 --conditions occluded --placements flat --orientations 1`: only the occluded 45° view, flat, one orientation (no normal bursts, no injected-pose test).
- `--reps 5 --retries-per-view 5`: 5 bursts at 45°, and **each** failed burst gets the full recovery sequence (same view, best angle, 90° CCW, overhead). So 5 reps = up to 5 recovery runs.
- `--retry --retry-best --retry-orbit`: turns on the steps. Overhead is on by default; `--orbit-stops` is 1 by default.
- Gripper mask and nearest-to-target filter are on by default.
- tau_c comes from the pipeline (gearbox 0.90, L 0.85, T 0.81, hemi 0.85), so no `--tau-c`. Gearbox uses the 153 mm STL by default.

## 1. Plan-only (no motion), once per object

Shows where each recovery view puts the camera. Use the `orbit ccw 90` TCP to see which side the second board must block in setup B.

```
python vgap_gate_eval_ur5e.py --object gearbox --color white --robot ur-rtde --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --plan-only
python vgap_gate_eval_ur5e.py --object l_block --color white --robot ur-rtde --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --plan-only
python vgap_gate_eval_ur5e.py --object t_block --color white --robot ur-rtde --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --plan-only
python vgap_gate_eval_ur5e.py --object hemisphere --color white --robot ur-rtde --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --retry --retry-best --retry-orbit --target-max-dist 0.10 --plan-only
```

## 2. Pilot: 1 rep per setup (gearbox)

```
python vgap_gate_eval_ur5e.py --object gearbox --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 1 --retries-per-view 1 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/pilot_A_gearbox_white
python vgap_gate_eval_ur5e.py --object gearbox --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 1 --retries-per-view 1 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/pilot_B_gearbox_white
```

Check `recovery.csv` in each folder: the row with `new_state` Valid should have `stage` = `orbit` for A and `overhead` for B. Also check the `dropped` column in `gate_eval.csv` (the gripper should show up there, not as a gearbox pose).
Keep the pilots out of the final report (they're in their own `pilot_` folders).

## 3. Setup A (one board): all 4 objects

```
python vgap_gate_eval_ur5e.py --object gearbox --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/A_gearbox_white
python vgap_gate_eval_ur5e.py --object l_block --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/A_l_block_white
python vgap_gate_eval_ur5e.py --object t_block --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/A_t_block_white
python vgap_gate_eval_ur5e.py --object hemisphere --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --out data/gate_eval_occl/A_hemisphere_white
```

## 4. Setup B (two-board corner): all 4 objects

```
python vgap_gate_eval_ur5e.py --object gearbox --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/B_gearbox_white
python vgap_gate_eval_ur5e.py --object l_block --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/B_l_block_white
python vgap_gate_eval_ur5e.py --object t_block --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --orbit-out 0.12 --out data/gate_eval_occl/B_t_block_white
python vgap_gate_eval_ur5e.py --object hemisphere --color white --robot ur-rtde --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --target 0.443,-0.559,0.146 --angles 45 --conditions occluded --placements flat --orientations 1 --reps 5 --retries-per-view 5 --retry --retry-best --retry-orbit --target-max-dist 0.10 --out data/gate_eval_occl/B_hemisphere_white
```

## 5. Report (setups A and B together; A_/B_ are in the folder names)

```
python vgap_gate_eval_ur5e.py --report data/gate_eval_occl/A_*/ data/gate_eval_occl/B_*/
```

Then copy all of `data/gate_eval_occl/` off the lab machine.

## Notes
- `--orbit-out 0.12` (gearbox, L, T): the 90 deg orbit view moves 12 cm farther out (-x here) at the same camera height, still aimed at the object (tilt 30 -> 42.7 deg), so it is not almost top-down. Not on the hemisphere: its orbit tilt is 45 deg, +12 cm would be 55 deg, above the 45 deg safety limit.
- Leave out `--no-confirm` for the first runs: the script waits for Enter before each move, so you can watch the orbit.
- Hemisphere: its best angle is 0°, so step 1 already goes overhead and there is no separate overhead step. It should recover at step 1 in both setups.
- The board has to block the 45° view from the `--side` direction (default `+y`, i.e. the side the camera tilts toward in plan-only).
