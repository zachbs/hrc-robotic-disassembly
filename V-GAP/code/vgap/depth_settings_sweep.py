"""RealSense laser / exposure sweep with no YOLO: just the point clouds.

1. A window shows the color image. Drag a box around the object and press
   ENTER (or SPACE). Press c to cancel.
2. The script steps through every laser power / exposure pair, captures a
   short burst for each, and keeps the depth inside your box.
3. It shows the results three ways:
   - a table in the console (valid-depth % inside the box, point count),
   - a grid of the depth images, one tile per setting, labelled,
   - one Open3D window with every point cloud laid out in the same grid
     (rows = laser power, columns = exposure, low to high from top left).
   Every cloud is also saved to sweep_<timestamp>/ as a .ply.

Run it once each for the white, grey and dark version of an object, then
pick ONE setting for all trials and put it in CAMERA_SETTINGS.

Usage:  python depth_settings_sweep.py
"""

import os
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

LASER_POWERS = [90.0, 150.0, 250.0, 360.0]        # mW   (D435 default 150)
EXPOSURES_US = [2000.0, 4000.0, 8500.0, 16000.0]  # us   (D435 default 8500)
WIDTH, HEIGHT, FPS = 640, 480, 30
BURST = 10            # frames per setting (median of valid readings)
SETTLE_FRAMES = 15    # frames dropped after changing a setting
MAX_DEPTH_M = 0.8
REMOVE_TABLE = True   # drop the dominant plane so only the object is left
VOXEL = 0.002


def median_ignore_zeros(frames):
  stack = np.stack(frames).astype(np.float32)
  stack[stack == 0] = np.nan
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    med = np.nanmedian(stack, axis=0)
  return np.nan_to_num(med).astype(np.uint16)


def roi_cloud(depth, roi, intr):
  x, y, w, h = roi
  crop = np.zeros_like(depth)
  crop[y:y + h, x:x + w] = depth[y:y + h, x:x + w]
  pcd = o3d.geometry.PointCloud.create_from_depth_image(
      o3d.geometry.Image(crop), intr, depth_scale=1000.0,
      depth_trunc=MAX_DEPTH_M)
  pcd = pcd.voxel_down_sample(VOXEL)
  if REMOVE_TABLE and len(pcd.points) > 100:
    _, inl = pcd.segment_plane(0.004, 3, 300)
    # only treat it as the table if it is a big share of the box
    if len(inl) > 0.3 * len(pcd.points):
      pcd = pcd.select_by_index(inl, invert=True)
  return pcd


def label_tile(depth, roi, text):
  x, y, w, h = roi
  d = depth[y:y + h, x:x + w].astype(np.float32)
  valid = d > 0
  vis = np.zeros((h, w, 3), np.uint8)
  if valid.any():
    lo, hi = np.percentile(d[valid], [2, 98])
    norm = np.clip((d - lo) / max(hi - lo, 1), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0  # holes are black
  vis = cv2.resize(vis, (240, int(240 * h / max(w, 1))))
  cv2.putText(vis, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
              (255, 255, 255), 1)
  return vis


def main():
  pipe = rs.pipeline()
  cfg = rs.config()
  cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
  cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
  prof = pipe.start(cfg)
  align = rs.align(rs.stream.color)
  ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  intr = o3d.camera.PinholeCameraIntrinsic(ci.width, ci.height, ci.fx, ci.fy,
                                           ci.ppx, ci.ppy)
  sensor = prof.get_device().first_depth_sensor()
  if sensor.supports(rs.option.enable_auto_exposure):
    sensor.set_option(rs.option.enable_auto_exposure, False)
  time.sleep(1)

  for _ in range(30):  # let auto white balance settle
    frames = align.process(pipe.wait_for_frames())
  color = np.asanyarray(frames.get_color_frame().get_data()).copy()
  roi = cv2.selectROI("Drag a box around the object, then ENTER", color,
                      showCrosshair=True)
  cv2.destroyAllWindows()
  if roi[2] == 0 or roi[3] == 0:
    print("[-] No box selected.")
    pipe.stop()
    return
  x, y, w, h = roi
  n_box = w * h

  out_dir = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  os.makedirs(out_dir, exist_ok=True)
  print(f"\n{'laser mW':>9} {'exp us':>8} {'valid % (1 frame)':>18} "
        f"{'valid % (burst)':>16} {'points':>7}")

  results, tiles = [], []
  try:
    for lp in LASER_POWERS:
      row_tiles = []
      for ex in EXPOSURES_US:
        if sensor.supports(rs.option.laser_power):
          sensor.set_option(rs.option.laser_power, lp)
        sensor.set_option(rs.option.exposure, ex)
        for _ in range(SETTLE_FRAMES):
          pipe.wait_for_frames()
        burst = []
        for _ in range(BURST):
          f = align.process(pipe.wait_for_frames()).get_depth_frame()
          burst.append(np.asanyarray(f.get_data()).copy())
        single = np.mean(burst[-1][y:y + h, x:x + w] > 0) * 100
        med = median_ignore_zeros(burst)
        merged = np.mean(med[y:y + h, x:x + w] > 0) * 100
        pcd = roi_cloud(med, roi, intr)
        o3d.io.write_point_cloud(
            os.path.join(out_dir, f"laser{lp:.0f}_exp{ex:.0f}.ply"), pcd)
        results.append(dict(laser=lp, exposure=ex, single=single,
                            merged=merged, pcd=pcd))
        row_tiles.append(label_tile(med, roi, f"L{lp:.0f} E{ex:.0f} {merged:.0f}%"))
        print(f"{lp:9.0f} {ex:8.0f} {single:18.1f} {merged:16.1f} "
              f"{len(pcd.points):7d}")
      tiles.append(row_tiles)
  finally:
    pipe.stop()

  best = max(results, key=lambda r: r["merged"])
  print(f"\n[+] Most depth in the box: laser {best['laser']:.0f} mW, exposure "
        f"{best['exposure']:.0f} us ({best['merged']:.1f}% valid, box {n_box} px)")
  print(f"[+] Point clouds saved in {out_dir}/")
  print("    (valid % counts the whole box, so background in it counts too;"
        " compare settings with each other, not to 100%)")

  # depth image grid
  th = max(t.shape[0] for r in tiles for t in r)
  grid = np.vstack([np.hstack([cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 2,
                                                  cv2.BORDER_CONSTANT)
                               for t in r]) for r in tiles])
  cv2.imshow("Depth per setting (rows: laser low->high, cols: exposure "
             "low->high). Any key = 3D view", grid)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # 3D grid, same layout as the image grid
  geoms, spacing = [], None
  for r in results:
    if len(r["pcd"].points):
      ext = r["pcd"].get_max_bound() - r["pcd"].get_min_bound()
      spacing = max(spacing or 0, ext[0] * 1.3, ext[1] * 1.3)
  spacing = spacing or 0.15
  for i, r in enumerate(results):
    row, col = divmod(i, len(EXPOSURES_US))
    p = o3d.geometry.PointCloud(r["pcd"])
    if len(p.points):
      p.translate(-p.get_center() + np.array([col * spacing, row * spacing, 0]))
      color = [0.1, 0.8, 0.1] if r is best else [1, 0.706, 0]
      p.paint_uniform_color(color)
      geoms.append(p)
  print("[*] 3D grid: rows = laser low->high (top to bottom), cols = exposure "
        "low->high (left to right); green = most depth.")
  o3d.visualization.draw_geometries(
      geoms, window_name="Point cloud per setting (green = most depth)",
      front=[0, 0, -1], up=[0, -1, 0])


if __name__ == "__main__":
  main()
