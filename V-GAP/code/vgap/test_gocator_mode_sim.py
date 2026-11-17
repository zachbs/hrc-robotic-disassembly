"""End-to-end test of vgap_gate_eval_ur5e --gocator with a simulated RealSense + Gocator."""
import sys, os, json, runpy, numpy as np
VG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VG)
os.chdir(sys.argv[1])
import calibrate_tau_c_ur5e as cal, vgap_trials_ur5e as tr, vgap_pipeline as vp
import sim_stress_test as st, gocator_gt as ggt, gocator_cam

obj = sys.argv[2] if len(sys.argv) > 2 else "l_block"
for k, rel in st.MESHES.items():
  vp.OBJECT_CONFIGS[k]["cad_path"] = os.path.join(st.ROOT, rel)
  vp.OBJECT_CONFIGS[k]["cad_units"] = "mm"
X = np.eye(4); X[:3, 3] = [0.0, 0.04, -0.107]
np.save("X.npy", X)
Xg_S = cal.from_ur_pose([0.0, -0.09, 0.02, 0.0, 0.0, 0.0])   # TCP -> sensor optics (z = view)
# Real Gocator output frame: origin ~mid-range (0.219 m out), Z toward the sensor.
T_G_S = np.diag([1.0, -1.0, -1.0, 1.0]); T_G_S[2, 3] = 0.219
Xg = Xg_S @ np.linalg.inv(T_G_S)                              # TCP -> Gocator output frame
np.save("Xg.npy", Xg)
spot = (T_G_S @ [0, 0, 0.22, 1])[:3]
json.dump(dict(center_gocator_m=spot.tolist(),
               tool_axis_gocator=(Xg[:3, :3].T @ [0, 0, 1.0]).tolist(),
               view_axis_gocator=[0, 0, -1], standoff_m=0.001, residual_mean_mm=0.3),
          open("Xg.json", "w"))
STATE = {}
class Cam(cal.SimCamera):
  def __init__(self, serial=None):
    super().__init__(obj, "white", 3, X)
    STATE["cam"] = self
cal.RealSenseCamera = Cam
class Robot(cal.DryRunRobot):
  def __init__(self, *a, **k):
    super().__init__(False, cal.from_ur_pose(tr.OBSERVE_POSE))
    STATE["robot"] = self
  def move_tcp(self, T, label=""):
    self.T = T.copy()
cal.URRtdeRobot = Robot
class Goc:
  def __init__(self, ip=None): pass
  def snapshot(self):
    cam, rob = STATE["cam"], STATE["robot"]
    T_w_s = rob.get_tcp() @ Xg
    cadm = ggt.CadTarget  # noqa
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
      cad = vp.load_cad_model(vp.OBJECT_CONFIGS[obj])
    return ggt._sim_scan(cad, cam.T_base_obj, T_w_s)
  def close(self): pass
gocator_cam.GocatorCamera = Goc
tr.wait_for_operator = lambda *a, **k: None
STATIC = len(sys.argv) > 3 and sys.argv[3] == "static"
if STATIC:
  # 60 deg view, TCP placed so the object sits at the Gocator spot (both sensors
  # see it from one pose), as the operator does by hand for the static 60 deg set
  import contextlib as _c, io as _io
  with _c.redirect_stdout(_io.StringIO()):
    c0 = cal.SimCamera(obj, "white", 3, X).object_center()
  R = tr.tool_rotation(60, list(tr.SIDES)[0])
  T_st = np.eye(4); T_st[:3, :3] = R
  T_st[:3, 3] = c0 - R @ (Xg[:3, :3] @ spot + Xg[:3, 3])
  STATE["robot"] = type("R", (), {"get_tcp": lambda self: T_st.copy()})()
  ARGS = ["--static", "--static-tcp", ",".join(f"{v:.6f}" for v in cal.to_ur_pose(T_st)),
          "--angles", "60", "--placements", "flat", "--orientations", "1", "--reps", "3",
          "--no-gui", "--gocator", "--gocator-hand-eye", "Xg.npy", "--gocator-scans", "2",
          "--out", "out_static_" + obj]
import contextlib as _c2, io as _io2
with _c2.redirect_stdout(_io2.StringIO()):
  _cam0 = cal.SimCamera(obj, "white", 3, X)
  _cad0 = vp.load_cad_model(vp.OBJECT_CONFIGS[obj])
_P = np.asarray(_cad0.dense_points) @ _cam0.T_base_obj[:3, :3].T + _cam0.T_base_obj[:3, 3]
TZ = ["--gocator-table-z", f"{_P[:, 2].min():.5f}"]       # sim table = lowest object point
sys.argv = ["x", "--object", obj, "--color", "white", "--robot", "ur-rtde", "--hand-eye", "X.npy",
            "--angles", "0,30", "--placements", "flat", "--orientations", "1", "--reps", "3",
            "--no-gui", "--gocator", "--gocator-hand-eye", "Xg.npy", "--gocator-scans", "2",
            "--min-tip-object", "0.05", "--out", "out_" + obj] + TZ
if STATIC:
  sys.argv = ["x", "--object", obj, "--color", "white", "--hand-eye", "X.npy"] + ARGS + TZ
runpy.run_path(os.path.join(VG, "vgap_gate_eval_ur5e.py"), run_name="__main__")
