# Simulated stress test of vgap_pipeline.py (2026-10-15)

Script: `code/vgap/sim_stress_test.py`. Raw per-trial data: `code/vgap/sim_stress_results/trials.csv`. Tables: `code/vgap/sim_stress_results/summary.txt`.
Pipeline: vgap_pipeline.py with the L/T class check. The gearbox CAD is the 153 mm STL, because `origin_centered_local_global.pcd` was not available for the simulation.

## Setup
- 480 trials: 4 objects x 3 colors x 4 poses x 5 views (0/15/30/45/60 deg) x 2 random yaws and azimuths, camera at 0.40 m.
- Poses: flat; propped on a 6 mm screw (tilt about 4-5 deg); propped on a 20 mm block (12-17 deg); flat with a 40x40x70 mm clutter box beside it.
- Depth: a 25-frame burst with z^2 noise, low-frequency noise and 1 mm quantization. Pixels at depth edges go invalid, flying pixels appear, and grazing angles drop out. Dropout also depends on color: the dark reflective print loses near-normal pixels to IR saturation. Each burst then goes through the pipeline's own zero-ignoring median and bilateral filter.
- RGB: shaded renders on a textured table. **The trained YOLO model at the time (`10-08-26seg26s.pt`, the base model before the no-null fine-tune) ran on every render.** When it missed or gave the wrong class (151 of 480), a YOLO-like mask was simulated instead (1/4-resolution prototype, -1 to +2 px boundary).
- A pose counts as a success at <= 5 deg (symmetry-aware) and <= 5 mm. It counts as bad at > 10 deg or > 10 mm, or with the wrong class.

## Headline
| | Accepted & correct | False accepts | Correctly rejected | Rejected but fine |
|---|---|---|---|---|
| Main path (RANSAC+ICP+hypotheses+gate) | 52% | 28 | 165 | 31 |
| With table yaw-sweep fallback | 82% | 28 | 12 | 40 |

Excluding the hemisphere propped on the block, false accepts are **2 of 450** (both L block, at the 15 deg view).
On accepted poses, median error is 0.2-0.6 deg and 0.7-1.8 mm. Color barely matters (80-84% for every color with the fallback), consistent with the exposure work.

## Findings
1. **Near-overhead views (0-15 deg) are where RANSAC fails** for the gearbox, L and T. The view shows mostly one flat face, and RANSAC fits it to the wrong face (e.g. gearbox side-on: 90-118 deg off, fitness 0.6 vs 1.0 at truth). The gate rejected every one of these, so the main path only accepts 28% at 15 deg vs 77% at 60 deg. The fallback rescued most of them: the gearbox goes from 53% to 90%, the L from 37% to 91%. **The real runs have the fallback switched on, so check `used_fallback` in the saved result.json files; real "no failures" may include fallback rescues.**
2. **Hemisphere propped up is a blind spot: 26 false accepts.** The hemisphere's axis is taken from the table plane, so when it sits on a block its error equals the prop tilt (16.6 deg). A tilted dome has almost the same silhouette, so C stays 0.86-0.99 and the gate passes it. On the screw (about 5 deg) the error is about 4.9 deg. Fix: estimate the axis from the object itself (the rim of the flat base, visible at tilt), not the table. At minimum, state the flat-resting assumption for the hemisphere.
3. **Clutter causes false rejections, not false accepts.** YOLO masks only the visible part, but the CAD projection is the full silhouette, so IoU drops on correct poses (33 "rejected but fine" with the fallback, mostly T block and gearbox). Fix: occlusion-aware projection, i.e. hide CAD pixels that lie behind measured depth.
4. **L block: 2 false accepts** at the 15 deg view: 17.5 deg / 12.7 mm off with C 0.886, just above tau_c 0.85. Calibrating tau_c on real data should decide whether 0.88-0.90 is better for L.
5. **Propped gearbox/L/T are fine.** The screw and block cases reach 93-100% with the fallback, because ICP corrects the tilt after the flat-assumption sweep.
6. **The class check never accepted a wrong class.** In the sim, YOLO called the L "gearbox" 38 times and "T" 19 times. Every wrong-class result ended as Conflict.
7. Median runtime is about 4 s per capture on 1 CPU core (L/T run two CADs).

## Caveats
Simulated, not hardware: CAD-perfect geometry, idealized table, no multipath, no real IR speckle, and no real calibration error. The 0-15 deg RANSAC weakness is consistent with the earlier simulation, but the hardware runs disagree, which is why item 1's `used_fallback` check matters.
