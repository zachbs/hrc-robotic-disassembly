# Pose method comparison in simulation (2026-10-15)

Script: `code/vgap/sim_method_compare.py`. Raw data: `code/vgap/sim_method_results/trials.csv`. All tables (by object, scene, view and color): `code/vgap/sim_method_results/summary.txt`.
Scenes are the same 360 gearbox/L/T scenes as the stress test (same seeds, depth model and masks). The hemisphere is left out because it has no corners or features. Every method's pose goes through the same gate (C >= tau_c). Only `current` also has the 90 deg hypotheses and the table fallback.

## Results (360 trials)
| Method | Correct before gate | Accepted & correct | False accepts | Fine but rejected | No pose | Median error (accepted) | Time |
|---|---|---|---|---|---|---|---|
| PCA only (previous group) | 0% | 0% | **168** | 5 | 0 | 3.1 deg / 14.2 mm | 0.2 s |
| PCA + ICP | 70% | 67% | 18 | 12 | 0 | 0.26 deg / 0.9 mm | 0.6 s |
| RANSAC + ICP (plain) | 38% | 35% | 0 | 13 | 0 | 0.23 deg / 0.7 mm | 2.0 s |
| **Current pipeline** | 96% | 87% | 1 | 36 | 0 | 0.39 deg / 1.1 mm | 2.7 s |
| SolvePnP + ICP* | 99% | 91% | 0 | 32 | 1 | 0.19 deg / 0.8 mm | 0.1 s |
| Feature constraints + rotate + ICP* | 84% | 79% | 0 | 19 | 54 | 0.22 deg / 0.9 mm | 0.4 s |

*These two use a **simulated keypoint detector**: 2 px noise, 5% outliers, true visibility, and labels consistent up to symmetry. A real detector (YOLO-pose) would be worse, especially on the gearbox's identical corners and in rooms it wasn't trained in. Treat these rows as an upper bound.

By object, accepted & correct:
| | PCA | PCA+ICP | RANSAC+ICP | Current | SolvePnP* | Feature* |
|---|---|---|---|---|---|---|
| Gearbox | 0% | 80% | 48% | 90% | 90% | 55% |
| L | 0% | 68% | 25% | 91% | 97% | 97% |
| T | 0% | 54% | 32% | 81% | 86% | 86% |

## What it says
1. **PCA alone never gives a usable pose.** The scan only sees the visible surface, so its centroid sits about 14 mm from the true centre. Worse, the gate **accepted 168 of these bad poses**: a 14 mm shift mostly along the camera ray barely changes the 2D silhouette. This is the strongest argument against PCA, and it is also a real gate limitation (see 5).
2. **ICP fixes most PCA translation error (67%),** but PCA's axes flip on near-square or partial views, giving 18 false accepts. Plain RANSAC+ICP is never accepted wrong, but it only gets 35% right (it fails at 0-15 deg views).
3. **The current pipeline is the best detector-free method:** 87% accepted and correct, 1 false accept. The hypotheses and fallback take plain RANSAC+ICP from 38% to 96% correct poses.
4. **Keypoint methods are as good or better *if* the keypoints are good.** SolvePnP reaches 91% and is 20-30x faster. The lip-feature method matches it on the L (97%) and T (86%), but on the gearbox it gets no pose in 49 of 120 trials: from 45-60 deg the far lip is hidden behind the body, so only one lip is visible. It works best at 0 deg (94%), which is exactly where RANSAC is weakest. So it complements RANSAC well as an overhead/near-overhead method.
5. **Gate limitation:** C is 2D, so it is weak against errors along the viewing ray (depth offsets). A cheap fix is to add a 3D check, e.g. the share of scan points within 3 mm of the posed CAD (already printed by `show_3d_overlap`), with its own threshold. That would have rejected the PCA poses.
6. Clutter is the main source of "fine but rejected" for every method. This is the occlusion issue from the stress test, not a method issue.

## For the paper
- Detector-free comparison: PCA (prior work) vs PCA+ICP vs RANSAC+ICP vs current, all through the same gate. This is fully supported by this simulation and needs no new training.
- If a keypoint model gets trained, SolvePnP+ICP is the stronger keypoint baseline. The lip-constraint method suits overhead views of the gearbox (both lips visible).
- State that the keypoint rows use an idealised detector unless real YOLO-pose results replace them.
