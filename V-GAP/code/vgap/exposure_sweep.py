"""Finds RealSense laser power / exposure that give the most depth on an object.

Put one object in view, run this, and it steps through laser power and depth
exposure combinations. For each one it captures a short burst and reports the
valid-depth ratio inside the YOLO mask (plan Eq. 13), plus how much of the
mask is saturated in the left IR image. Run it on the white, grey and dark
version of an object, then pick ONE setting for all trials (the plan keeps
sensor settings fixed) and put it in CAMERA_SETTINGS in vgap_pipeline.py.

Usage:  python exposure_sweep.py
"""

import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from vgap_pipeline import (CAMERA_SETTINGS, YOLO_MODEL_PATH,
                           _best_known_detection, temporal_median_ignore_zeros)

LASER_POWERS = [90.0, 150.0, 250.0, 360.0]          # mW (D435 default 150)
EXPOSURES_US = [2000.0, 4000.0, 8500.0, 16000.0]     # us (D435 default 8500)
BURST = 10
SETTLE_FRAMES = 15


def main():
  yolo = YOLO(YOLO_MODEL_PATH)
  s = CAMERA_SETTINGS
  pipe = rs.pipeline()
  cfg = rs.config()
  cfg.enable_stream(rs.stream.color, s["width"], s["height"], rs.format.bgr8, s["fps"])
  cfg.enable_stream(rs.stream.depth, s["width"], s["height"], rs.format.z16, s["fps"])
  cfg.enable_stream(rs.stream.infrared, 1, s["width"], s["height"], rs.format.y8, s["fps"])
  prof = pipe.start(cfg)
  align = rs.align(rs.stream.color)
  sensor = prof.get_device().first_depth_sensor()
  sensor.set_option(rs.option.enable_auto_exposure, False)
  time.sleep(1)

  # one YOLO mask from a color frame (the object does not move)
  frames = align.process(pipe.wait_for_frames())
  color = np.asanyarray(frames.get_color_frame().get_data()).copy()
  res = yolo(color, conf=s["yolo_conf"], verbose=False)[0]
  idx = _best_known_detection(res)
  if idx is None or res.masks is None:
    print("[-] No object detected; place one object in view and retry.")
    pipe.stop()
    return
  h, w = color.shape[:2]
  mask = cv2.resize(res.masks.data[idx].cpu().numpy(), (w, h),
                    interpolation=cv2.INTER_NEAREST) > 0.5
  name = res.names[int(res.boxes.cls[idx])]
  print(f"[+] Object: {name}, mask {mask.sum()} px\n")
  print(f"{'laser mW':>9} {'exp us':>8} {'valid %':>8} {'median-burst %':>15} "
        f"{'IR saturated %':>15}")

  rows = []
  for lp in LASER_POWERS:
    for ex in EXPOSURES_US:
      sensor.set_option(rs.option.laser_power, lp)
      sensor.set_option(rs.option.exposure, ex)
      for _ in range(SETTLE_FRAMES):
        pipe.wait_for_frames()
      burst, sat = [], []
      for _ in range(BURST):
        raw = pipe.wait_for_frames()
        ir = np.asanyarray(raw.get_infrared_frame(1).get_data())
        f = align.process(raw)
        burst.append(np.asanyarray(f.get_depth_frame().get_data()).copy())
        sat.append(np.mean(ir >= 250))  # IR is not aligned; whole-frame share
      single = np.mean(burst[-1][mask] > 0) * 100
      med = temporal_median_ignore_zeros(burst)
      merged = np.mean(med[mask] > 0) * 100
      rows.append((lp, ex, single, merged, np.mean(sat) * 100))
      print(f"{lp:9.0f} {ex:8.0f} {single:8.1f} {merged:15.1f} {rows[-1][4]:15.2f}")

  best = max(rows, key=lambda r: r[3])
  print(f"\n[+] Best for {name}: laser {best[0]:.0f} mW, exposure {best[1]:.0f} us "
        f"({best[3]:.1f}% valid after burst median)")
  pipe.stop()


if __name__ == "__main__":
  main()
