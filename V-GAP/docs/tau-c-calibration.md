# Calibrating tau_c per object (2026-10-16)

Scripts: `code/vgap/calibrate_tau_c_ur5e.py` (collects data), `code/vgap/recommend_tau_c.py` (picks tau_c).

## Method
1. Per object and color, the UR5e moves the wrist camera to tilts 0/15/30/45/60 deg x azimuths 0/90/180/270 at 0.40 m (20 views) and runs the normal pipeline at each view. C, state, margin, class, fallback, etc. go to a CSV.
2. Ground truth: you approve one pose by eye at the 45 deg view. With the TCP pose and the hand-eye matrix, that pose becomes the truth for every other view (the object must not move). Correct = <= 5 deg and <= 5 mm. Wrong = wrong class, or > 10 deg or > 10 mm. Without hand-eye, `--label manual` asks y/n at every view.
3. Injected errors: your pipeline almost never fails on its own, so at each correct view the true pose is pushed off (yaw 15/30/90/180, flip, 12/20 mm shifts, tilt, depth shift) and run through ICP. Wrong poses that ICP leaves behind are realistic local minima.
4. tau_c = the lowest threshold with zero false accepts among wrong poses that only C can catch (margin, class and hemisphere-axis checks already reject the others). That also keeps the most correct poses.

## Simulated starting values (replace with real data)
| Object | tau_c now | Sim tau_c | Correct accepted at sim tau_c | Notes |
|---|---|---|---|---|
| Gearbox | 0.80 | 0.81 | 98% | 2 false accepts at 0.80 |
| L block | 0.85 | 0.895 | 85% (97% at 0.85) | set by the known 15 deg view case (17.5 deg off, C 0.886) |
| T block | 0.85 | 0.865 | 95% | 1 false accept at 0.85 |
| Hemisphere | 0.85 | 0.835 | 95% | no ICP, so set by raw shifted poses; tilt is handled by the axis check |
Data: 480-trial stress test, hemisphere axis-fix rerun, and a 12-run sim of the calibration script (240 views + 3,000 injected poses).

## Findings
- **C is blind to depth and tilt errors.** A pose 15 mm too far from the camera, or tilted 15 deg, scores C 0.90-0.98, the same as a correct one. No tau_c fixes that. ICP corrects these as long as it starts within its threshold, but a 3D depth check (scan points within x mm of the posed CAD) is the right gate for them. This matches the earlier PCA-only finding.
- Gearbox: a 15 deg yaw error with no ICP passes about 80% of the time, because the box is nearly square (1.24:1). ICP fixed every 15 and 30 deg yaw injection in the sim.
- Color barely changes tau_c, except one grey L case. Use one value per object.
- In the sim, the reference label matched the true label in 237 of 240 views. The 3 misses were "borderline" calls caused by a 3.6 mm error in the reference pose. Pick the reference view carefully.

## Hand-eye (added 2026-10-16)
The original `T_flange_cam.npy` is a hand measurement: identity rotation, camera 35 mm along flange -y, 0 mm along z. A few cm of hand-eye error mislabels almost every view; in the sim, a 10 cm error labeled 17 of 20 good views "wrong". So the sweep now solves the hand-eye itself (AX = XB, Park-Martin, RANSAC over views, own implementation because OpenCV 5 dropped `calibrateHandEye` from the main module). Sim, starting from that file with a 106 mm / 3 deg error: recovered to 0.8-2.1 mm and 0.1 deg (L, T, gearbox), and the labels matched the truth in 59 of 60 views. The hemisphere can't solve it (yaw unobservable), so it reuses the matrix from an L or T run (`--no-refine-hand-eye`), which gave 19 of 19 correct.
The lab setup (from loop_pickup_with_adaptive_gripper.py / semester_demo.py): UR IP 192.168.1.5, wrist RealSense serial <CAMERA_SERIAL> (a second overview camera is also connected), T_base_cam = T_base_tcp @ T_flange_cam, OBSERVE_POSE near (0.20, -0.51, 0.50).

## Trial runner (added 2026-10-16, replaces the 360 deg sweep for real runs)
The 20-view sweep collided with itself on the real UR5e. The operator asked for semester_demo-style motion instead: `code/vgap_trials_ur5e.py`. The robot finds the object from OBSERVE_POSE and tilts in one vertical plane (tool rotvec (pi - tilt, 0, 0), like the demo's 45 deg orbit; the wrist never winds). The angles are 0/30/45/60; 75 and 90 put the flange within 5 cm of the table or below it at 0.4 m. At each angle the operator turns the object to 4 orientations (hemisphere 1), and each view gets 10 bursts. Labels: one y/n per distinct pose, and the repeats are labelled by distance (the camera doesn't move, so no hand-eye accuracy is needed). Outputs: captures.csv (all metrics), summary.txt / summary_by_view.csv, calib_*.csv for `recommend_tau_c.py --sweep-weights` (best wM and tau_c). Sim run, L block grey, 160 bursts: pass 100%, labels within truth (max true error 6 deg), repeatability 0.06-1.8 mm (worst at 0 deg).

## Not run on hardware
The real-robot path (`--robot ur-rtde`, ur_rtde moveL at 5 cm/s, Enter before each move) and the hands-free RealSense capture have not been run. Run `--plan-only` first and check the poses on the pendant.
