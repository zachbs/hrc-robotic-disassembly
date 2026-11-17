# Grasp trials: V-GAP vs plain RANSAC+ICP (UR5e + Robotiq 2F)

Script: `code/vgap_grasp_trials_ur5e.py`. Put it in the same folder as `vgap_pipeline.py` on the lab PC and run every command from that folder, like the gate eval.

## What a trial does
1. You place the part on the taped spot, flat or propped, at the yaw the script asks for.
2. The arm goes to the view (0, 30 or 45 deg) and takes bursts.
   * **vgap**: up to 2 bursts. The first Valid pose is grasped. If no pose is Valid, there is NO grasp (`no_grasp_gate`).
   * **plain**: the RANSAC+ICP pose for YOLO's class, before the gate's checks. It is always grasped. The script also logs whether the gate would have accepted it.
3. The grasp is planned from the CAD. A preview is saved to `trials/tNNN_*/preview.png`: green = mask, red = CAD, blue = fingers.
4. The arm moves hover, then pre-grasp (8 cm back), then a **guarded descent** at 2 cm/s. The descent stops if the wrist force goes over 25 N.
5. The gripper closes. The part is lifted 10 cm and held 5 s. Then you answer y / s / n (s = held, but it shifted in the fingers).
6. A held part is put back where it was and released.

**Success rule (fixed before the trials):** the part is lifted and still held after the 5 s hold. Every refusal, collision, abort or protective stop counts as a failure for that method.

**Design:** 5 setups x 2 reps x 2 methods = 20 trials per part. That is 10 V-GAP and 10 plain grasps per part, on the same placements and views. Objects and setups are shuffled, and so is the method order in each rep. The plan is saved first in `plan.csv`.

Default setups (`--setups`, placement:yaw:view tilt): flat 0 deg @ 30, flat 90 @ 45, flat 180 @ 0, propped 0 @ 30, propped 90 @ 45. The yaw is how far you turn the part from your 0 mark. The hemisphere ignores yaw.

## Grasp points (CAD frame; check with --mode check-grasps)
| Part | Grasp | Why |
|---|---|---|
| L | across one arm (33.3 mm), 15 mm past the inner corner | Center of mass (-13.3, -13.3) is in the notch, so it can't be gripped. Gripped 15 mm out along the arm so the fingers clear the other arm by ~18 mm (pilot at 3 mm came within ~3 mm). Both arms are listed, and the script picks the one with the least wrist turn. |
| T | across the stem (33.3 mm) at y = -12 | ~15 mm clearance below the crossbar, 25 mm from the center of mass. Backup: crossbar ends. |
| gearbox | 2F-140: across the 90 mm body at the center, 25 mm deep | Center of mass, 20 mm slack per side. **2F-85 can't do this (85 < 90 mm)**. Its only option is an end-lip pinch, with 2.5 mm finger clearance; that is off unless `--allow-experimental`. |
| hemisphere | 2F-140: across the dome 8 mm above the base, any yaw | 98.7 mm chord. The contact is nearly vertical, so the part won't squeeze out. **2F-85: no grasp** (100 mm > 85 mm). |

So with a **2F-85** the trial is L + T only (40 trials, ~1.5 h). With a **2F-140** it covers all 4 parts (80 trials, ~3 h).

## Before you start
* Tape an outline of each part at the spot under the observe pose, with a "0" yaw mark. Prop the part on the same screw as in the gate runs.
* Use the same TCP (0, 0, 160 mm) as the gate runs. The script reads TCP poses over RTDE.
* The Robotiq URCap must be installed (the script talks to it on port 63352). If that doesn't work, use `--gripper-io manual`, and you open and close the gripper on the pendant when asked.

## Commands (white parts; replace 2f85 with 2f140 if that's your gripper)
```bash
# 0. offline: clearances for your gripper (no robot)
python vgap_grasp_trials_ur5e.py --mode check-grasps --gripper 2f85

# 1. once: fingertip offset + finger closing axis (freedrive touch on the table, then a 20 mm test move)
#    writes data/grasp_trials/setup.json; redo it if the TCP or fingers change
python vgap_grasp_trials_ur5e.py --mode setup --gripper 2f85 --table-z 0.008

# 2. pilot: 2 trials on the L, Enter before every move, watch with a hand on the stop
python vgap_grasp_trials_ur5e.py --gripper 2f85 --serial <CAMERA_SERIAL> \
    --hand-eye matrix_config/T_flange_cam_updated_zfix.npy \
    --objects l_block --setups flat:0:30 --reps 1 --out data/grasp_trials/pilot

# 3. full run (Enter only before each descent once the pilot looks right)
python vgap_grasp_trials_ur5e.py --gripper 2f85 --serial <CAMERA_SERIAL> \
    --hand-eye matrix_config/T_flange_cam_updated_zfix.npy --confirm grasp

# 4. after a crash / protective stop / break: continue the same plan
python vgap_grasp_trials_ur5e.py --gripper 2f85 --serial <CAMERA_SERIAL> \
    --hand-eye matrix_config/T_flange_cam_updated_zfix.npy --confirm grasp \
    --resume data/grasp_trials/<run folder>

# 5. summary (also written at the end of each run)
python vgap_grasp_trials_ur5e.py --mode report --report "data/grasp_trials/2f*/trials.csv"
```

## While running
* `q` at any Enter prompt stops. Then choose `a` (abort for safety, logged as a failure), `s` (skip, redo it with --resume) or `x` (exit).
* Abort only for safety, such as a move that would hit the table or a fixture. Never abort because the pose "looks wrong", because that would bias the plain method.
* **Protective stop:** the trial is logged as `protective_stop`. Clear it on the pendant and press Enter. The script reconnects and goes on.
* If the guarded descent touches something, the trial is logged as `fail_collision` and the arm backs off.

## Outputs (`data/grasp_trials/<gripper>_white_<time>/`)
`plan.csv`, `trials.csv` (one row per trial: outcome, gate state, C, F3, plain pose's C/F3/would-pass, grasp used, Robotiq object status and finger width after the close and after the hold, max force), `trials/tNNN_*/preview.png` + saved captures, `grasp_report.txt`, `run_config.json`.

The report has success per trial and per attempt (with 95% CI), failed attempts, a paired McNemar test, and success of plain grasps split by whether the gate would have accepted the pose.

## Notes / limits
* Finger sizes are approximate, from the Robotiq drawings. They are used only for the clearance checks.
* `--table-z 0.008` (Gocator table). The fingertips stay at least 6 mm above it, otherwise the grasp is made shallower (to no less than 8 mm of finger depth).
* Tested in simulation only (flat placements, both grippers, full loop). UNTESTED ON HARDWARE.
