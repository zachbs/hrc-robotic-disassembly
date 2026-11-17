"""Collect hard examples for YOLO retraining.

Live RealSense + YOLO view. Press a key to pick the object in front of the
camera; from then on, any frame where YOLO's confidence for that class is
below CONF_SAVE (or the class is missing, or another class wins) is saved as
a clean color image into hard_examples/<class>/<color>/.

Keys:
  1 = L-shape   2 = T-shape   3 = hemisphere   4 = gearbox
  c = next filament color (white -> grey -> reflective)
  0 = stop saving (pause)
  s = save the current frame now (manual)
  q / Esc = quit

The banner at the top shows the count for each color of the selected class
(counted from files on disk, so it carries over between runs) and turns green
when a color reaches TARGET_PER_COLOR.

Saved images have no overlay, so they can go straight into labeling.
"""
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ==============================================================================
# SETTINGS (camera values match CAMERA_SETTINGS in vgap_pipeline.py)
# ==============================================================================
YOLO_MODEL_PATH = "../Python Files Disassembly/runs/segment/yolo26s_ft_nonull/weights/best.pt"   # deployed no-null model
WIDTH, HEIGHT, FPS = 640, 480, 30
LASER_POWER = 150.0
DEPTH_EXPOSURE_US = 4000.0

CONF_SAVE = 0.80        # save when the selected class's confidence is below this
CONF_DETECT = 0.10      # YOLO's own cutoff; low so weak detections still show
SAVE_INTERVAL_S = 0.5   # at most 2 saves per second (avoids near-duplicates)
OUT_DIR = "hard_examples"
COLORS = ["white", "grey", "reflective"]
TARGET_PER_COLOR = 30

# Key -> (folder name, keys used to match YOLO class names)
KEY_TO_CLASS = {
    ord("1"): "l_block",
    ord("2"): "t_block",
    ord("3"): "hemisphere",
    ord("4"): "gearbox",
}

# Same aliases as vgap_pipeline.py, so YOLO's class names map to the folders
CLASS_ALIASES = {
    "gearbox": "gearbox", "gear_box": "gearbox",
    "l_block": "l_block", "l_shape": "l_block", "l_shaped": "l_block",
    "l": "l_block",
    "t_block": "t_block", "t_shape": "t_block", "t_shaped": "t_block",
    "t": "t_block",
    "hemisphere": "hemisphere", "dome": "hemisphere", "hemi": "hemisphere",
    "sphere": "hemisphere", "semi_sphere": "hemisphere",
}


def object_key(class_name):
  k = class_name.strip().lower().replace("-", "_").replace(" ", "_")
  return CLASS_ALIASES.get(k)


def start_realsense():
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
  config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
  profile = pipeline.start(config)
  depth_sensor = profile.get_device().first_depth_sensor()
  if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, LASER_POWER)
  if depth_sensor.supports(rs.option.enable_auto_exposure):
    depth_sensor.set_option(rs.option.enable_auto_exposure, False)
  if depth_sensor.supports(rs.option.exposure):
    depth_sensor.set_option(rs.option.exposure, DEPTH_EXPOSURE_US)
  time.sleep(1)
  return pipeline


def analyze(result, target):
  """Returns (target_conf, top_key, top_conf) for one YOLO result.

  target_conf is the best confidence of the selected class (0 if missing);
  top_key / top_conf is the most confident detection of any class.
  """
  target_conf, top_key, top_conf = 0.0, None, 0.0
  if result.boxes is None:
    return target_conf, top_key, top_conf
  for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
    key = object_key(result.names[int(cls)])
    if key == target:
      target_conf = max(target_conf, conf)
    if conf > top_conf:
      top_key, top_conf = key or result.names[int(cls)], conf
  return target_conf, top_key, top_conf


def count_on_disk(target, filament):
  folder = os.path.join(OUT_DIR, target, filament)
  if not os.path.isdir(folder):
    return 0
  return sum(f.lower().endswith(".png") for f in os.listdir(folder))


def draw_banner(view, target, filament, counts):
  """Top bar: class, current color, and count/target for each color."""
  cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
  if target is None:
    cv2.putText(view, f"PAUSED | color: {filament} (c to change)", (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return
  x = 10
  cv2.putText(view, target, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              (255, 255, 255), 2)
  x += 110
  for c in COLORS:
    n = counts[(target, c)]
    text = f"{c[:5]} {n}/{TARGET_PER_COLOR}"  # "refle" keeps the bar on screen
    col = (0, 200, 0) if n >= TARGET_PER_COLOR else (0, 165, 255)
    if c == filament:
      text = f"[{text}]"
    cv2.putText(view, text, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    x += 165


def save_frame(color, target, filament, reason, target_conf):
  folder = os.path.join(OUT_DIR, target, filament)
  os.makedirs(folder, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
  name = (f"{target}_{filament}_{stamp}_{reason}"
          f"_c{int(round(target_conf * 100)):02d}.png")
  path = os.path.join(folder, name)
  cv2.imwrite(path, color)
  return path


def main():
  model = YOLO(YOLO_MODEL_PATH)
  print(f"[*] YOLO classes: {model.names}")
  unmapped = [n for n in model.names.values() if object_key(n) is None]
  if unmapped:
    print(f"[!] No folder mapping for YOLO classes {unmapped}; "
          f"add them to CLASS_ALIASES")

  pipeline = start_realsense()
  target = None
  filament = COLORS[0]
  last_save = 0.0
  counts = {(k, c): count_on_disk(k, c)
            for k in KEY_TO_CLASS.values() for c in COLORS}
  print("[*] Press 1=L  2=T  3=hemisphere  4=gearbox  c=color  0=pause  "
        "s=save  q=quit")

  try:
    while True:
      frames = pipeline.wait_for_frames()
      color_frame = frames.get_color_frame()
      if not color_frame:
        continue
      color = np.asanyarray(color_frame.get_data()).copy()  # clean copy to save

      result = model.predict(color, conf=CONF_DETECT, verbose=False)[0]
      view = result.plot()  # overlay for display only

      status, colour = "press 1/2/3/4 to start", (200, 200, 200)
      if target is not None:
        target_conf, top_key, top_conf = analyze(result, target)
        if target_conf == 0.0:
          reason = "missed" if top_key is None else f"as_{top_key}"
        elif top_key != target:
          reason = f"as_{top_key}"
        elif target_conf < CONF_SAVE:
          reason = "lowconf"
        else:
          reason = None

        status = (f"{target}: conf {target_conf:.2f}"
                  f"{'' if top_key in (None, target) else f' | top {top_key} {top_conf:.2f}'}"
                  f" | {filament}")
        colour = (0, 200, 0) if reason is None else (0, 0, 255)

        now = time.time()
        if reason and now - last_save >= SAVE_INTERVAL_S:
          path = save_frame(color, target, filament, reason, target_conf)
          counts[(target, filament)] += 1
          last_save = now
          print(f"[+] {reason:>14}  {path}")

      draw_banner(view, target, filament, counts)
      cv2.putText(view, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                  (0, 0, 0), 4)
      cv2.putText(view, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                  colour, 2)
      cv2.imshow("Hard example collector", view)

      key = cv2.waitKey(1) & 0xFF
      if key in (ord("q"), 27):
        break
      if key in KEY_TO_CLASS:
        target = KEY_TO_CLASS[key]
        print(f"[*] Selected {target}")
      elif key == ord("c"):
        filament = COLORS[(COLORS.index(filament) + 1) % len(COLORS)]
        print(f"[*] Color {filament}")
      elif key == ord("0"):
        target = None
        print("[*] Paused")
      elif key == ord("s") and target is not None:
        tc = analyze(result, target)[0]
        path = save_frame(color, target, filament, "manual", tc)
        counts[(target, filament)] += 1
        print(f"[+] {'manual':>14}  {path}")
  finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[*] Images on disk per class/color:")
    for (k, c), n in counts.items():
      print(f"    {k:>10} {c:>10}: {n}")


if __name__ == "__main__":
  main()
