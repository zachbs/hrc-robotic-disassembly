# Grasp trials: V-GAP vs plain RANSAC+ICP (UR5e + Robotiq 2F-140, white parts, 2026-11-02)

Run: `2f140_white_20261102_203300/` (80 trials, seed 69580). Pilots (`pilot*`) are excluded.
Redone trials: 9-12 (the gripper fingers were detected as a "gearbox" before the target filter was added) and 41 (the gearbox was still on the table at the start of the hemisphere block). The report uses the newest row of each trial.
Setup: fingertip center in TCP frame (-1.7, -75.2, 131) mm, measured twice (fit residual 0.1-0.3 mm). Fingers close along tool x.
Design: 4 parts x 5 setups (flat yaw 0/90/180 at 0/30/45 deg views, propped at 30/45 deg) x 2 reps x 2 methods, paired on the same placement and view, in random order.
Success rule (set before the trials): the part is lifted 10 cm and still held after a 5 s hold. A refused grasp, a safety abort or a gate rejection counts as a failure.

## Main result (success per trial)
| Part | V-GAP | Plain RANSAC+ICP | Paired (V-GAP only / plain only) | McNemar exact p |
|---|---|---|---|---|
| L | 10/10 | 2/10 | 8 / 0 | 0.0078 |
| T | 10/10 | 1/10 | 9 / 0 | 0.0039 |
| Gearbox | 9/10 | 2/10 | 7 / 0 | 0.016 |
| Hemisphere | 10/10 | 10/10 | 0 / 0 | 1 |
| **All** | **39/40 = 97.5% [87-100]** | **15/40 = 37.5% [24-53]** | **24 / 0** | **1.2e-7** |

95% Wilson CIs in brackets. Without the hemisphere: V-GAP 29/30 vs plain 5/30.

## How the methods failed
* **V-GAP:** 1 gate rejection (gearbox propped at 30 deg, C 0.892 < tau_c 0.90, 2 bursts) and **0 failed grasp attempts**. Every grasp it attempted held (39/39).
* **Plain, 25 failures:**
  * 16 `no_grasp_unsafe`: the pose had the part's top face sideways (tipped 80-120 deg). That needs a side approach, so the planner refused it (L 8, gearbox 8).
  * 8 `aborted_safety`, all T: the plain pose had the T flipped (crossbar where the stem is), so the fingers would have closed diagonally across the part. The operator stopped these at the descent prompt to avoid damaging the printed T. Objective support for these aborts:
    * Each plain pose was 28-124 deg / 19-29 mm from V-GAP's pose.
    * The gate would have rejected all 8.
    * The saved preview.png of each trial shows the fingers off the part.
  * 1 `fail_dropped` (T).
* Plain poses differed from V-GAP's by 28-125 deg / 19-47 mm in all 25 failures. The plain pose matched V-GAP's (0 deg / 0 mm) for every hemisphere trial, for 2 gearbox trials and for 3 T trials.

## The gate predicts plain grasp outcome
Of the 34 plain poses where the gate could be evaluated (YOLO class = final class):
* gate would ACCEPT: **13/13 grasps succeeded**
* gate would REJECT: **0/21 succeeded**

## Sensitivity
* Excluding the 8 operator safety aborts: plain 15/32 (46.9%). Paired: 16 V-GAP-only vs 0, p = 3.1e-5.
* Per attempted grasp (aborts counted as attempts): V-GAP 39/39, plain 15/24.
* Peak wrist force during descents was 6.4 N (stop limit 25 N): no collisions and no protective stops.
* The L/T class check fixed 4 V-GAP trials (YOLO said T on the L: 7, 10, 12, 20). Plain uses YOLO's class. It grasped the L as a T twice (17, 19), and the T-stem grasp still held.
* The Robotiq reported object detected (OBJ 2) on every success. The width readout is a linear estimate, so don't quote widths.

## Summary
Grasp success: V-GAP 97.5% vs plain 37.5% on the same 40 paired placements (McNemar p < 1e-6). Zero failed grasp attempts with V-GAP vs 9 physical failures (8 aborts + 1 drop) and 16 infeasible poses with plain.

Limitation: the operator was not blind to the method. The 8 T aborts were operator calls, made to avoid part damage. With those aborts excluded the result still holds (p = 3e-5). One color (white), one gripper, 10 grasps per part per method.
