"""Gocator hand-eye: T_flange_goc (Gocator frame -> TCP frame), like T_flange_cam.

Put a sphere-shaped target on the table and leave it there: the hemisphere
print (default, R = 50 mm, flat side down; the sphere center is the middle
of its base) or a matte ball (--radius). Then jog the arm (freedrive or
pendant) to 10-12 different poses where the Gocator sees the target at about
16-27 cm, and press Enter at each one. Vary BOTH:
  - the tilt (0 to ~35 deg, from different sides), and
  - where the target sits in the view (left/right/top/bottom of the image,
    near and far). If it is always in the middle at the same distance, the
    sensor's rotation cannot be solved (the residual still looks fine!).

Each capture: TCP pose (read-only RTDE, the arm is never commanded), one
Gocator snapshot, table removed, sphere fitted -> center in the Gocator
frame. Solve: the sphere center is one fixed point in the base frame,
T_base_tcp_i @ X @ c_i = p for all i (linear first guess, then least squares).

  python gocator_handeye.py --robot-ip 192.168.1.5
      Enter = capture, u = undo last, q = solve and save
  python gocator_handeye.py --solve data/handeye/gocator_captures.npz   (re-solve)
  python gocator_handeye.py --sim                                       (self-test)

Output: data/handeye/T_flange_goc.npy (+ .json: residuals, view axis, standoff).
Residual = how far each capture's sphere center lands from the common point;
under ~0.5 mm mean is good (the RealSense ArUco hand-eye was 1.5 mm).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def from_ur_pose(p):
  T = np.eye(4)
  T[:3, :3], _ = cv2.Rodrigues(np.asarray(p[3:], float))
  T[:3, 3] = p[:3]
  return T


def solve(Ts, cs):
  """Ts: base->TCP poses (4x4), cs: sphere centers in the Gocator frame.
  Returns X (TCP -> Gocator... i.e. maps Gocator points into the TCP frame),
  the fixed point p (base) and per-capture residuals (m)."""
  A, b = [], []
  for T, c in zip(Ts, cs):
    R, q = T[:3, :3], T[:3, 3]
    for r in range(3):
      A.append(np.concatenate([np.kron(R[r], c), R[r], -np.eye(3)[r]]))
      b.append(-q[r])
  sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
  U, _, Vt = np.linalg.svd(sol[:9].reshape(3, 3))
  R0 = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
  x0 = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), sol[9:12], sol[12:15]])

  def res(x):
    R = cv2.Rodrigues(x[:3])[0]
    return np.concatenate([T[:3, :3] @ (R @ c + x[3:6]) + T[:3, 3] - x[6:9]
                           for T, c in zip(Ts, cs)])
  x = least_squares(res, x0, loss="soft_l1", f_scale=0.001).x
  X = np.eye(4)
  X[:3, :3], X[:3, 3] = cv2.Rodrigues(x[:3])[0], x[3:6]
  r = np.linalg.norm(res(x).reshape(-1, 3), axis=1)
  return X, x[6:9], r


def rotation_spread_deg(Ts):
  ang = [np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))
         for i, a in enumerate(Ts) for b in Ts[i + 1:]]
  return max(ang) if ang else 0.0


def save(X, p, r, cs, Ts, out):
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  np.save(out, X)
  dirs = np.array([c / np.linalg.norm(c) for c in cs])
  v = dirs.mean(0)
  info = dict(n_captures=len(cs), residual_mean_mm=round(1000 * float(r.mean()), 3),
              residual_max_mm=round(1000 * float(r.max()), 3),
              rotation_spread_deg=round(rotation_spread_deg(Ts), 1),
              fixed_point_base_m=[round(float(v_), 5) for v_ in p],
              # Gocator-frame origin is not at the sensor (near mid-range), so the
              # gate eval uses the average target spot + the tool axis instead.
              center_gocator_m=[round(float(v_), 5) for v_ in np.mean(cs, 0)],
              tool_axis_gocator=[round(float(v_), 5) for v_ in X[:3, :3].T @ [0, 0, 1.0]],
              view_axis_gocator=[round(float(v_), 5) for v_ in v / np.linalg.norm(v)],
              standoff_m=round(float(np.mean([np.linalg.norm(c) for c in cs])), 4))
  with open(out.replace(".npy", ".json"), "w") as f:
    json.dump(info, f, indent=2)
  print(json.dumps(info, indent=2))
  print(f"[+] saved {out}")


def sphere_center(pts, radius):
  import gocator_gt as gg
  import vgap_pipeline as vp
  obj, _ = gg.isolate_object(pts)
  c, n_in = vp._fit_sphere_known_radius(obj, radius, 0.001, 3000, np.random.default_rng(0))
  return c, n_in, len(obj)


def sim_test():
  rng = np.random.default_rng(0)
  X_true = from_ur_pose([0.02, -0.06, 0.05, 0.1, -0.05, 1.5])
  p = np.array([0.45, -0.55, 0.12])
  Ts, cs = [], []
  for _ in range(10):
    # tool pointing roughly down at p, random tilt/azimuth, sensor ~0.22 m away
    tilt, az = np.radians(rng.uniform(0, 35)), rng.uniform(0, 2 * np.pi)
    R = cv2.Rodrigues(np.array([np.sin(az), -np.cos(az), 0]) * tilt)[0] @ \
        cv2.Rodrigues(np.array([np.pi, 0, 0]))[0] @ cv2.Rodrigues(np.array([0, 0, rng.uniform(-1, 1)]))[0]
    T = np.eye(4)
    T[:3, :3] = R
    # the target anywhere in the view: +-40 mm sideways, 170-260 mm away
    c_i = np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(0.17, 0.26)])
    T[:3, 3] = p - R @ (X_true[:3, :3] @ c_i + X_true[:3, 3])
    c = np.linalg.inv(X_true) @ np.linalg.inv(T) @ np.append(p, 1)
    Ts.append(T)
    cs.append(c[:3] + rng.normal(0, 0.0002, 3))       # 0.2 mm sphere-fit noise
  X, p_est, r = solve(Ts, cs)
  e_t = 1000 * np.linalg.norm(X[:3, 3] - X_true[:3, 3])
  e_r = np.degrees(np.arccos(np.clip((np.trace(X[:3, :3].T @ X_true[:3, :3]) - 1) / 2, -1, 1)))
  print(f"sim: X error {e_t:.2f} mm / {e_r:.3f} deg, fixed point error "
        f"{1000 * np.linalg.norm(p_est - p):.2f} mm, residual mean {1000 * r.mean():.2f} mm")
  return e_t < 1.5 and e_r < 0.5


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--robot-ip", default="192.168.1.5")
  ap.add_argument("--gocator-ip", default="192.168.1.10")
  ap.add_argument("--radius", type=float, default=0.050, help="sphere radius, m")
  ap.add_argument("--out", default="data/handeye/T_flange_goc.npy")
  ap.add_argument("--solve", default=None, help="re-solve saved captures (.npz)")
  ap.add_argument("--sim", action="store_true")
  a = ap.parse_args()
  if a.sim:
    print("OK" if sim_test() else "FAILED")
    return
  if a.solve:
    d = np.load(a.solve)
    Ts, cs = list(d["Ts"]), list(d["cs"])
    if "zflip" not in d.files:        # captured before gocator_cam flipped z (11/01)
      print("[*] old capture file: flipping z of the sphere centers to the corrected frame")
      cs = [c * [1, 1, -1] for c in cs]
  else:
    import rtde_receive
    from gocator_cam import GocatorCamera
    rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
    g = GocatorCamera(a.gocator_ip)
    npz = os.path.join(os.path.dirname(a.out) or ".", "gocator_captures.npz")
    os.makedirs(os.path.dirname(npz) or ".", exist_ok=True)
    Ts, cs = [], []
    try:
      while True:
        k = input(f"[{len(Ts)} captures] jog to a new pose; Enter = capture, "
                  "u = undo, q = solve: ").strip().lower()
        if k == "q":
          break
        if k == "u":
          if Ts:
            Ts.pop(), cs.pop()
          continue
        # 11/01: the RTDE receive stream froze after 3 captures (same TCP pose for
        # every later capture -> 15 mm residual). Reconnect if needed and refuse
        # a pose that did not change since the last capture.
        try:
          if not rr.isConnected():
            rr.reconnect()
        except Exception:
          rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
        T = from_ur_pose(rr.getActualTCPPose())
        if Ts and np.linalg.norm(T[:3, 3] - Ts[-1][:3, 3]) < 0.002 and \
            rotation_spread_deg([T, Ts[-1]]) < 1.0:
          print("  [-] the arm pose is the same as the last capture (did not move, or the "
                "robot stream froze). Move further, or restart the script if it repeats.")
          rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
          continue
        print(f"  TCP {np.round(1000 * T[:3, 3], 1)} mm")
        try:
          c, n_in, n_obj = sphere_center(g.snapshot(), a.radius)
        except Exception as e:
          print(f"  [-] capture failed: {e}")
          continue
        print(f"  sphere center (Gocator frame) {np.round(1000 * c, 1)} mm, "
              f"{n_in}/{n_obj} points on the sphere")
        Ts.append(T)
        cs.append(c)
        np.savez(npz, Ts=np.array(Ts), cs=np.array(cs), zflip=True)
        if len(Ts) >= 4:
          _, _, r = solve(Ts, cs)
          print(f"  running residual: mean {1000 * r.mean():.2f} mm, max {1000 * r.max():.2f} mm")
    finally:
      g.close()
  if len(Ts) < 6:
    raise SystemExit(f"only {len(Ts)} captures; take at least 6 (8-12 is better)")
  if rotation_spread_deg(Ts) < 20:
    print("[!] the poses differ by less than 20 deg; tilt more for a reliable rotation")
  spread = np.ptp(np.array(cs), axis=0) * 1000
  if min(spread) < 30:
    print(f"[!] the target covered only {np.round(spread)} mm of the view (x, y, z); "
          "move it to the edges and near/far, or the rotation is unreliable")
  X, p, r = solve(Ts, cs)
  for i, ri in enumerate(r):
    print(f"  capture {i + 1}: {1000 * ri:.2f} mm")
  save(X, p, r, cs, Ts, a.out)


if __name__ == "__main__":
  main()
