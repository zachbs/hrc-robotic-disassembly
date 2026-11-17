# Static 60° gate tests (arm never connected or moved)

Copy the new `vgap_gate_eval_ur5e.py` to the Jetson first.

Before you start: jog the arm to your 60° view with the pendant, then leave it. With `--static` the script never connects to the UR5e and never moves it (it raises an error if anything tries). `--angles 60` is only the label in the CSV.

Each run: 4 flat positions, then 4 propped positions (on the screw), 5 bursts each = 40 bursts per object/color, plus the injected-pose test at each position. The script prompts you before each new position (orientation 2/4 says "turn it 90 deg"; any new position is fine, just don't move it during the 10 bursts).

Gripper mask is on. The target filter is off (no `--target`), so it keeps the most confident detection, as in the main runs. Hemisphere uses `--axis-ambig-deg 15` (validated 10/26).

Optional: add `--static-tcp x,y,z,rx,ry,rz` (TCP pose shown on the pendant) so the base-frame columns are real. Without it they are relative to the TCP. Gate results don't use them.

## White
```
python vgap_gate_eval_ur5e.py --object gearbox --color white --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/gearbox_white
python vgap_gate_eval_ur5e.py --object l_block --color white --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/l_block_white
python vgap_gate_eval_ur5e.py --object t_block --color white --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/t_block_white
python vgap_gate_eval_ur5e.py --object hemisphere --color white --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --axis-ambig-deg 15 --out data/gate_eval_static60/hemisphere_white
```

## Grey
```
python vgap_gate_eval_ur5e.py --object gearbox --color grey --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/gearbox_grey
python vgap_gate_eval_ur5e.py --object l_block --color grey --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/l_block_grey
python vgap_gate_eval_ur5e.py --object t_block --color grey --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/t_block_grey
python vgap_gate_eval_ur5e.py --object hemisphere --color grey --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --axis-ambig-deg 15 --out data/gate_eval_static60/hemisphere_grey
```

## Reflective
```
python vgap_gate_eval_ur5e.py --object gearbox --color reflective --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/gearbox_reflective
python vgap_gate_eval_ur5e.py --object l_block --color reflective --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/l_block_reflective
python vgap_gate_eval_ur5e.py --object t_block --color reflective --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --out data/gate_eval_static60/t_block_reflective
python vgap_gate_eval_ur5e.py --object hemisphere --color reflective --static --angles 60 --serial <CAMERA_SERIAL> --hand-eye matrix_config/T_flange_cam.npy --placements flat,propped --orientations 4 --reps 5 --conditions normal --axis-ambig-deg 15 --out data/gate_eval_static60/hemisphere_reflective
```

## Report (all 12 runs together)
```
python vgap_gate_eval_ur5e.py --report data/gate_eval_static60/*/ --out data/gate_eval_static60
```

Back up `data/gate_eval_static60/` off the lab machine when done.
