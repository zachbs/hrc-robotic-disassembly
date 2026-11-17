# Method notes and command logs

Working notes written during the experiments (Oct-Nov 2026). Each file records what was run, the exact commands, and what was found.

| File | Topic |
|---|---|
| `pipeline-parameters.md` | Per-object pipeline settings and the first offline test |
| `tau-c-calibration.md` | How the gate threshold tau_c was calibrated per object on the UR5e |
| `gate-evaluation.md` | Design of the gate evaluation runs (outcomes, injected wrong poses, report sections) |
| `static-60deg-commands.md` | 60 degree static-arm test |
| `occlusion-test-commands.md` | Occlusion / recovery test (setups A and B) |
| `gocator-ground-truth.md` | Gocator 3210 ground truth: method, calibration, validity rules, regrades |
| `grasp-trial-commands.md` | Paired grasp trials, V-GAP vs plain RANSAC+ICP |

Notes on paths:
- The commands were run on the lab Jetson from the folder that holds the scripts, `cad/` and `matrix_config/` (`Python Files Disassembly/`). In this repo the scripts are in `V-GAP/code/vgap/`; run them from `V-GAP/` as `python code/vgap/<script>.py ...`.
- `matrix_config/` (camera hand-eye matrices) stays in `Python Files Disassembly/matrix_config/`, so pass `--hand-eye "../Python Files Disassembly/matrix_config/<file>.npy"`.
- Some notes mention the earlier YOLO model `10-08-26seg26s.pt`. The final experiments used the no-null fine-tune (`runs/segment/yolo26s_ft_nonull/weights/best.pt`).
