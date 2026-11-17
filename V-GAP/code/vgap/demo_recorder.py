"""Conference demo video for vgap_gate_eval_ur5e.py (--demo-video).

Records one continuous real-time MP4 while the gate eval runs, laid out as a
split screen (1920x1080 by default):

  +---------------------------------------------------------------+
  | V-GAP title | current step (first view / step 0-3) | clock     |
  +-------------------------------------+-------------------------+
  |                                     | 3D CHECK                |
  |  LIVE: RealSense RGB + YOLO26 masks | posed CAD (blue) on the |
  |  (every detection, also the ones    | point cloud, slowly     |
  |   the gripper / target filters drop)| turning; scan points    |
  |                                     | green <= 3 mm, red more |
  |                                     +-------------------------+
  |                                     | 2D GATE                 |
  |                                     | YOLO mask (green) vs    |
  |                                     | projected CAD (magenta) |
  |                                     | IB, IM, C vs tau_c      |
  +-------------------------------------+-------------------------+
  | VALID - pose released  /  CONFLICT (reason) - recovering        |
  | class, C, F3, margin, accepted / rejected tally                 |
  +---------------------------------------------------------------+

The live panel never stops (frames come from the gate eval's own camera
reads, and from a background read when the main loop is idle, e.g. while the
arm moves). The two right panels hold the result of the latest burst until
the next one. The clock in the top bar is wall time, so the phone video can
be synced to it (show the screen or say the time once on the phone video);
<video>_events.jsonl has the same timestamps for every burst and step.

Nothing here changes what the gate eval computes. It runs a second YOLO copy
on the live stream, so per-burst runtime goes up a little: do not use demo
runs for runtime numbers or mix them into the paper data folders.
"""
import copy
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

BG = (24, 24, 24)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
GREEN = (80, 210, 80)
RED = (60, 60, 235)
ORANGE = (0, 165, 255)
MAGENTA = (220, 60, 220)
CAD_BLUE = (255, 190, 40)
CLASS_COLORS = [(0, 140, 255), (255, 140, 0), (200, 80, 200), (80, 200, 80),
                (0, 220, 220), (200, 200, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.7, color=WHITE, thick=2):
  cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
  cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _fit(img, w, h):
  """Resize keeping aspect ratio, centered on a BG canvas of w x h."""
  ih, iw = img.shape[:2]
  s = min(w / iw, h / ih)
  r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))))
  out = np.full((h, w, 3), BG, np.uint8)
  y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
  out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
  return out


def _rot(axis, deg):
  a = np.radians(deg)
  c, s = np.cos(a), np.sin(a)
  if axis == "x":
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # about y (image vertical)


class DemoRecorder:
  def __init__(self, path, cam, robot=None, object_key="", fps=12, size=(1920, 1080),
               live_yolo=True, yolo_conf=None, tau_c=None, f3_min=None):
    self.path, self.cam, self.fps = path, cam, float(fps)
    self.W, self.H = size
    self.object_key, self.tau_c, self.f3_min = object_key, tau_c, f3_min
    self.cam_lock = threading.Lock()
    self.ev_lock = threading.Lock()
    self.frame, self.frame_t = None, 0.0
    self.stage, self.status = "Starting", ""
    self.res = None                       # latest burst result, prepared for drawing
    self.tally = dict(valid=0, conflict=0, none=0)
    self._stop = threading.Event()
    self.n_written = 0
    self.conflict_text = "recovery"       # what happens on a Conflict (grasp demo: no grasp)
    self.live_paused = ""                 # text shown instead of live YOLO while set

    # --- tap the camera: every frame the gate eval reads is also ours -----
    self._orig_frames = cam._frames

    def tapped():
      with self.cam_lock:
        d, c = self._orig_frames()
      self._feed(c)
      return d, c
    cam._frames = tapped

    # --- show arm moves in the status line --------------------------------
    if robot is not None and hasattr(robot, "move_tcp"):
      orig_move = robot.move_tcp

      def move(T, label=""):
        self.set_status(f"arm moving: {label}" if label else "arm moving")
        self.event("move", label=label)
        try:
          return orig_move(T, label)
        finally:
          self.set_status("")
      robot.move_tcp = move

    # --- second YOLO copy for the live panel (the predictor is not thread-safe)
    self.yolo = None
    if live_yolo and getattr(cam, "yolo", None) is not None:
      from ultralytics import YOLO
      w = getattr(cam.yolo, "ckpt_path", None)
      if not w:
        import vgap_pipeline as vp
        w = vp.YOLO_MODEL_PATH
      self.yolo = YOLO(w)
      settings = getattr(cam, "settings", {}) or {}
      self.yolo_conf = yolo_conf if yolo_conf is not None else settings.get("yolo_conf", 0.4)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    self.writer, self.path = self._open_writer(path)
    self.events_f = open(os.path.splitext(self.path)[0] + "_events.jsonl", "w")
    self.event("start", object=object_key, fps=self.fps, size=[self.W, self.H])
    self.t0 = time.time()
    self.thread = threading.Thread(target=self._loop, daemon=True)
    self.thread.start()
    print(f"[demo] recording {self.path} ({self.W}x{self.H} @ {self.fps:g} fps)")

  # ------------------------------------------------------------------ output
  def _open_writer(self, path):
    tries = [(path if path.endswith(".mp4") else path + ".mp4", "mp4v"),
             (os.path.splitext(path)[0] + ".avi", "MJPG")]
    for p, cc in tries:
      w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*cc), self.fps, (self.W, self.H))
      if w.isOpened():
        if cc != "mp4v":
          print(f"[demo] no MP4 encoder in this OpenCV build; writing {p} (MJPG, larger)")
        return w, p
    raise RuntimeError("could not open a video writer (mp4v or MJPG)")

  def event(self, kind, **kw):
    rec = dict(t=datetime.now().isoformat(timespec="milliseconds"),
               video_s=round(self.n_written / self.fps, 2), kind=kind, stage=self.stage, **kw)
    with self.ev_lock:
      self.events_f.write(json.dumps(rec, default=str) + "\n")
      self.events_f.flush()

  def set_stage(self, text):
    self.stage = text
    self.event("stage")

  def set_status(self, text):
    self.status = text

  def _feed(self, color):
    self.frame, self.frame_t = color, time.time()

  # ------------------------------------------------------------ burst result
  def on_burst(self, obs, out, cad):
    """Call after each burst (obs None = no detection)."""
    if obs is None or out is None:
      self.tally["none"] += 1
      self.res = dict(state="no detection", stage=self.stage, t=time.time())
      self.event("burst", state="no detection")
      return
    state = out.get("state", "?")
    if state != "plain":
      self.tally["valid" if state == "Valid" else "conflict"] += 1
    r = dict(state=state, reason=out.get("reason", ""), stage=self.stage, t=time.time(),
             cls=out.get("final_object", obs.object_key), yolo_cls=obs.object_key,
             conf=obs.confidence, C=out.get("C"), IB=out.get("IB"), IM=out.get("IM"),
             F3=out.get("F3"), margin=out.get("margin"),
             swapped=out.get("final_object", obs.object_key) != obs.object_key)
    r["gate_img"] = self._gate_panel(obs, out)
    r["pts3d"] = self._prep_3d(out, cad)
    self.res = r
    self.event("burst", state=state, reason=r["reason"], C=r["C"], F3=r["F3"],
               cls=r["cls"], yolo_cls=r["yolo_cls"])

  def _gate_panel(self, obs, out):
    img = (obs.color.astype(np.float32) * 0.45).astype(np.uint8)
    M = obs.mask.astype(bool)
    Mp = out.get("Mproj")
    Mp = np.zeros_like(M) if Mp is None else Mp.astype(bool)
    over = img.copy()
    over[M & ~Mp] = GREEN
    over[Mp & ~M] = MAGENTA
    over[M & Mp] = (235, 235, 235)
    img = cv2.addWeighted(img, 0.35, over, 0.65, 0)
    if obs.box is not None:
      b = obs.box
      cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), GREEN, 2)
    Bp = out.get("Bproj")
    if Bp is not None:
      cv2.rectangle(img, (Bp[0], Bp[1]), (Bp[2], Bp[3]), MAGENTA, 2)
    return img

  def _prep_3d(self, out, cad):
    scan, T = out.get("scan"), out.get("T_cad_to_cam")
    if scan is None or T is None or cad is None or len(scan.points) == 0:
      return None
    rng = np.random.default_rng(0)
    posed = copy.deepcopy(cad.pcd)
    posed.transform(T)
    S = np.asarray(scan.points)
    C = np.asarray(posed.points)
    d = np.asarray(scan.compute_point_cloud_distance(posed))
    if len(S) > 7000:
      i = rng.choice(len(S), 7000, replace=False)
      S, d = S[i], d[i]
    if len(C) > 7000:
      C = C[rng.choice(len(C), 7000, replace=False)]
    cen = S.mean(0)
    rad = max(np.linalg.norm(np.vstack([S, C]) - cen, axis=1).max(), 0.02)
    col_s = np.where((d <= 0.003)[:, None], GREEN, RED).astype(np.uint8)
    col_c = np.tile(np.array(CAD_BLUE, np.uint8), (len(C), 1))
    return dict(P=np.vstack([C, S]), col=np.vstack([col_c, col_s]), cen=cen, rad=rad)

  def _render_3d(self, p3, w, h, t):
    img = np.full((h, w, 3), (36, 36, 36), np.uint8)
    yaw = 40.0 * np.sin(2 * np.pi * t / 8.0)          # swing +-40 deg every 8 s
    R = _rot("x", -15.0) @ _rot("y", yaw)
    Q = (p3["P"] - p3["cen"]) @ R.T
    D = 3.0 * p3["rad"]
    f = 0.42 * min(w, h) * D / p3["rad"]
    z = Q[:, 2] + D
    u = (f * Q[:, 0] / z + w / 2).astype(int)
    v = (f * Q[:, 1] / z + h / 2).astype(int)
    order = np.argsort(-z)                              # far first, near last wins
    u, v, col = u[order], v[order], p3["col"][order]
    for du in (-1, 0, 1):
      for dv in (-1, 0, 1):
        uu, vv = u + du, v + dv
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        img[vv[ok], uu[ok]] = col[ok]
    return img

  # ------------------------------------------------------------- live panel
  def _live(self, frame):
    img = frame.copy()
    if self.live_paused:                  # grasp demo: pose locked, part close to the camera
      _text(img, self.live_paused, (12, img.shape[0] - 14), 0.6, WHITE, 2)
      return img
    if self.yolo is None:
      return img
    try:
      res = self.yolo(frame, conf=self.yolo_conf, agnostic_nms=True, verbose=False)[0]
    except Exception:
      return img
    if res.boxes is None or len(res.boxes) == 0:
      return img
    over = img.copy()
    polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
    for k in range(len(res.boxes)):
      ci = int(res.boxes.cls[k])
      col = CLASS_COLORS[ci % len(CLASS_COLORS)]
      if polys[k] is not None and len(polys[k]) >= 3:
        cv2.fillPoly(over, [polys[k].astype(np.int32)], col)
    img = cv2.addWeighted(img, 0.55, over, 0.45, 0)
    for k in range(len(res.boxes)):
      ci = int(res.boxes.cls[k])
      col = CLASS_COLORS[ci % len(CLASS_COLORS)]
      x1, y1, x2, y2 = [int(q) for q in res.boxes.xyxy[k].cpu().numpy()]
      cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
      _text(img, f"{res.names[ci]} {float(res.boxes.conf[k]):.2f}", (x1, max(18, y1 - 6)),
            0.55, col, 1)
    return img

  # --------------------------------------------------------------- compose
  def _compose(self, live, now):
    W, H = self.W, self.H
    top, bot = int(0.065 * H), int(0.10 * H)
    mid = H - top - bot
    lw = int(W * 0.625)
    rw = W - lw
    ph = mid // 2
    can = np.full((H, W, 3), BG, np.uint8)

    # top bar
    _text(can, "V-GAP: verify every pose before the robot uses it", (16, int(top * 0.68)),
          0.85, WHITE, 2)
    clock = datetime.now().strftime("%H:%M:%S.%f")[:-5]
    _text(can, clock, (W - 190, int(top * 0.68)), 0.8, GREY, 2)
    _text(can, self.stage[:60], (int(W * 0.45), int(top * 0.68)), 0.85, (0, 220, 255), 2)

    # live panel
    lv = _fit(live, lw, mid)
    _text(lv, "LIVE  RealSense RGB + YOLO26 segmentation", (14, 30), 0.7, WHITE, 2)
    if self.status:
      _text(lv, self.status[:70], (14, mid - 18), 0.7, ORANGE, 2)
    can[top:top + mid, :lw] = lv

    r = self.res
    # 3D panel
    if r is not None and r.get("pts3d") is not None:
      p3 = self._render_3d(r["pts3d"], rw, ph, now - self.t0)
    else:
      p3 = np.full((ph, rw, 3), (36, 36, 36), np.uint8)
      _text(p3, "no pose yet" if r is None else "no pose (no detection)",
            (20, ph // 2), 0.7, GREY, 2)
    _text(p3, "3D CHECK  CAD (blue) on point cloud", (12, 26), 0.6, WHITE, 1)
    _text(p3, "scan: green <= 3 mm from CAD, red farther", (12, 50), 0.5, GREY, 1)
    if r is not None and r.get("F3") is not None:
      lim = f" (min {self.f3_min:g})" if self.f3_min else ""
      _text(p3, f"F3 = {r['F3']:.2f}{lim}", (12, ph - 14), 0.7, WHITE, 2)
    can[top:top + ph, lw:] = p3

    # 2D gate panel
    if r is not None and r.get("gate_img") is not None:
      g = _fit(r["gate_img"], rw, mid - ph)
    else:
      g = np.full((mid - ph, rw, 3), (36, 36, 36), np.uint8)
    _text(g, "2D GATE  YOLO mask (green) vs projected CAD (magenta)", (12, 26), 0.55, WHITE, 1)
    if r is not None and r.get("C") is not None:
      tau = f"  tau_c {self.tau_c:g}" if self.tau_c else ""
      _text(g, f"IB {r['IB']:.2f}  IM {r['IM']:.2f}  C = 0.3 IB + 0.7 IM = {r['C']:.2f}{tau}",
            (12, mid - ph - 14), 0.6, WHITE, 2)
    can[top + ph:top + mid, lw:] = g
    cv2.line(can, (lw, top), (lw, top + mid), (70, 70, 70), 2)
    cv2.line(can, (lw, top + ph), (W, top + ph), (70, 70, 70), 2)

    # bottom verdict bar
    y1 = top + mid
    if r is None:
      verdict, col, line2 = "waiting for the first burst", GREY, ""
    elif r["state"] == "no detection":
      verdict, col, line2 = "NO DETECTION  part not visible from this view", ORANGE, \
          f"[{r['stage']}]"
    else:
      ok = r["state"] == "Valid"
      verdict = "VALID  pose released to the robot" if ok else \
          f"CONFLICT ({r['reason'] or 'gate'})  pose withheld, {self.conflict_text}"
      col = GREEN if ok else RED
      if r["state"] == "plain":          # grasp demo: plain RANSAC+ICP, no gate
        verdict, col = "NO GATE  plain pose sent as is", ORANGE
      cls = r["cls"] + (f" (YOLO said {r['yolo_cls']}, class check fixed it)"
                        if r["swapped"] else f" (YOLO {r['conf']:.2f})")
      line2 = f"[{r['stage']}]  class {cls}"
      if r.get("margin") is not None:
        line2 += f"  |  hypothesis margin {r['margin']:.2f}"
    cv2.rectangle(can, (0, y1), (W, H), (40, 40, 40), -1)
    _text(can, verdict, (16, y1 + int(bot * 0.45)), 1.05, col, 3)
    _text(can, line2[:120], (16, y1 + int(bot * 0.85)), 0.6, GREY, 1)
    t = self.tally
    _text(can, f"accepted {t['valid']}   rejected {t['conflict']}   no detection {t['none']}",
          (W - 560, y1 + int(bot * 0.45)), 0.65, WHITE, 2)
    return can

  # ------------------------------------------------------------------ loop
  def _loop(self):
    period = 1.0 / self.fps
    while not self._stop.is_set():
      tick = time.time()
      if tick - self.frame_t > 0.15 and self.cam_lock.acquire(blocking=False):
        try:                              # main loop idle (arm moving): read ourselves
          _, c = self._orig_frames()
          self._feed(c)
        except Exception:
          pass
        finally:
          self.cam_lock.release()
      if self.frame is None:
        time.sleep(0.02)
        continue
      try:
        can = self._compose(self._live(self.frame), tick)
      except Exception as e:              # never let the demo stop the experiment
        print(f"[demo] frame skipped: {e}")
        time.sleep(period)
        continue
      due = int((time.time() - self.t0) * self.fps) + 1
      while self.n_written < due:         # real-time video: repeat frames if slow
        self.writer.write(can)
        self.n_written += 1
      time.sleep(max(0.0, period - (time.time() - tick)))

  def close(self):
    self._stop.set()
    self.thread.join(timeout=5)
    self.cam._frames = self._orig_frames
    self.writer.release()
    self.event("end", frames=self.n_written)
    self.events_f.close()
    print(f"[demo] saved {self.path} ({self.n_written / self.fps:.0f} s)")

