"""Shift the camera hand-eye translation by a correction measured in the TCP frame.
Usage: python shift_hand_eye.py matrix_config/T_flange_cam_updated.npy 0 0 -6.3
(dx dy dz in mm = the V-GAP-minus-Gocator offset expressed in the TCP frame; it is
SUBTRACTED from the camera position). Writes <name>_zfix.npy next to the input
and never overwrites the original."""
import sys, numpy as np
src = sys.argv[1]
d = np.array([float(v) for v in sys.argv[2:5]]) / 1000.0
X = np.load(src).astype(float)
Xn = X.copy()
Xn[:3, 3] = X[:3, 3] - d
dst = src.replace(".npy", "_zfix.npy")
np.save(dst, Xn)
print("old t (mm):", np.round(1000 * X[:3, 3], 2))
print("new t (mm):", np.round(1000 * Xn[:3, 3], 2))
print("saved", dst)
