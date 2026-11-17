# vgap_pipeline.py: per-object parameters and offline test

Code: `code/vgap/vgap_pipeline.py` (new file; `IOU_pipeline.py` is untouched). Offline test: `code/vgap/test_vgap_pipeline_sim.py`, results in `test_vgap_pipeline_sim_results.txt`.

## Object sizes used

| Object | Size (mm) | Smallest feature that matters | CAD in config |
|---|---|---|---|
| Gearbox (assembled) | 153 x 90 x 64.8, lip to lip | lips: 9 thick, 20.7 out | your `captured_scans/origin_centered_local_global.pcd` |
| L block | 100 x 100 x 26.7, arms 33.3 | arm width 33.3, height 26.7 | `cad/l_block/l_block_100mm_reference.stl` * |
| T block | 100 x 100 x 26.7, bar and stem 33.3 | same | `cad/t_block/t_block_100mm_reference.stl` * |
| Hemisphere | D 100 x 50 | none (smooth) | `cad/hemisphere/hemisphere_100mm_reference.stl` * |

\* Built from the plan's Figure 1 dimensions scaled by 2/3. Swap in your own STLs. `load_cad_model()` warns if a CAD's size is more than 5% off the expected extent.

## Parameters and why

Rules of thumb: a D435 at 0.4 to 0.5 m has about 1 mm depth noise and 0.75 mm per pixel. Voxel is about 1/50 of the longest side and at least 3 to 4 voxels across the thinnest feature. Normals use 2 to 2.5x voxel, FPFH 5x voxel (kept below the smallest distinctive feature), RANSAC distance 1.5x voxel, and ICP runs about 5x, then 3x, then 1.5x voxel.

| Parameter | Gearbox | L / T block | Hemisphere |
|---|---|---|---|
| voxel | 2.5 mm (your tuned value) | 2.0 mm | 2.0 mm |
| mask erode | 2 px | 2 px | 2 px |
| outlier removal | nb 20, std 2.0 (protects lips) | nb 20, std 2.0 | nb 20, std 2.0 |
| normal radius | 5 mm | 5 mm | 6 mm |
| FPFH radius | 12.5 mm (lip scale) | 10 mm (inside the 33 mm arm) | not used |
| global method | FPFH RANSAC, 3.75 mm, 1M iters | FPFH RANSAC, 3 mm, 1M iters | known-radius sphere fit (R 50 mm, 2 mm inliers) + table normal |
| ICP thresholds | 12, 8, 4 mm | 10, 6, 3 mm | skipped (sphere fit is the least-squares fit) |
| ICP estimator | plane, plane, point | point, point, point | none |
| symmetry | auto: D2, 4 poses | auto: 180 deg about diagonal (L) or stem (T) | axial (yaw unobservable) |
| hypotheses tried | 90 deg yaw | 90, 180, 270 deg yaw | none |
| fallback | table yaw sweep | table yaw sweep | none |
| wB / wM / tau_c / delta | 0.3 / 0.7 / 0.85 / 0.10 | same | same |

Why the custom parts:
- **Hemisphere:** every patch of a sphere has the same FPFH, so feature RANSAC can't work. The sphere center is the base center, and the axis is the table normal measured from the depth ring around the mask.
- **L and T use point-to-point ICP:** from overhead the scan is a single flat face, and point-to-plane ICP can slide and turn freely within that plane. Point-to-point locks onto the outline.
- **Table yaw sweep fallback:** runs only when the first answer is a Conflict. It puts the object flat on the measured table at the mask centroid, sweeps yaw in 15 deg steps scored by C, then runs ICP on the best 3. It uses the 2D mask to choose the 3D pose, so turn it off (`fallback=None`) for the plan's "static view" baseline runs. Each result also records `used_fallback`.

## Offline test (simulated scans)

Each object sat on a table, viewed from 0.40 m at 0, 30 and 45 deg tilt, with 2 yaws and 2 runs each (12 runs per object). The test used 0.7 mm depth noise and the true silhouette as the YOLO mask. There was no IR dropout, so expect real data to be worse.

| Object | First pass (RANSAC+ICP) | Final | Gate |
|---|---|---|---|
| Gearbox | the 0 deg views needed the fallback | 12/12 within 3.3 deg and 2.4 mm | 12 Valid |
| L block | RANSAC failed in most views (flat faces give few distinct features) | 12/12 within 3.4 deg, 0.8 to 4.7 mm | 10 Valid, 2 Conflict (overhead, 4.7 mm slide, C 0.83) |
| T block | same as L | 12/12 within 1.8 deg, 0.6 to 6.6 mm | 10 Valid, 2 Conflict (overhead, 6.6 mm slide, C 0.81) |
| Hemisphere | sphere fit | 12/12 at 0.0 deg tilt error, 0.6 mm | 12 Valid |

Without the fallback (first run of the test), RANSAC+ICP alone failed in 23 of the 36 gearbox, L and T runs. The gate flagged every one of those failures, with one exception: an L block at 18 deg error scored C 0.86 and was accepted.

Takeaways:
- **Overhead (0 deg) is the hardest view for the three flat objects,** even though it gives the best depth. The plan's Eq. 12 moves toward overhead when U_live is high, so this tradeoff is worth a paragraph in the paper.
- The 2 remaining Conflicts are correct calls: the pose is 5 to 7 mm off in-plane, about the plan's smallest injected error.
- The hemisphere result is idealized, since a perfect sphere with clean depth makes the fit trivial. Its real test is the dark and reflective prints.
