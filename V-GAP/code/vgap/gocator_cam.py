"""Gocator 3210 snapshot -> numpy point cloud (sensor frame, meters, z flipped to be right-handed).

Thin wrapper around your Gocator.py / GoSdk_MsgHandler.py setup (same SDK
calls, same uniform-surface message), but it keeps one connection open and
returns the points instead of saving a PLY. Put this file next to
Gocator.py and GoSdk_MsgHandler.py (GO_SDK_4 must point at the SDK, as
gocator_scan.py sets it).

  python gocator_cam.py              one snapshot -> gocator_test.ply + point count
  python gocator_cam.py --n 5        5 snapshots (repeatability of the raw data)

The sensor job must output Uniform Surface data (the one Gocator.py saves).
"""
import argparse
import ctypes
import os
import time
from ctypes import byref

import numpy as np

SDK_PATH = "/home/robotics/Documents/human_robot_collab/test/fall_practice/GO_SDK"
os.environ.setdefault("GO_SDK_4", SDK_PATH)

GOCATOR_IP = "192.168.1.10"


class GocatorCamera:
  def __init__(self, ip=GOCATOR_IP, timeout_ms=20000):
    import Gocator as G          # loads the SDK libraries
    self.G = G
    self.Sdk, self.kApi = G.GoSdk, G.kApi
    self.Sdk.GoDataSet_Count.argtypes = [ctypes.c_void_p]
    self.Sdk.GoDataSet_Count.restype = ctypes.c_size_t
    self.Sdk.GoDataSet_At.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    self.Sdk.GoDataSet_At.restype = ctypes.c_void_p
    self.Sdk.GoSystem_ReceiveData.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_uint64]   # timeout in us
    self.Sdk.GoSystem_ReceiveData.restype = ctypes.c_int
    self.timeout_us = int(timeout_ms) * 1000
    self.last_info = {}
    self.api = ctypes.c_void_p()
    self.system = ctypes.c_void_p()
    self.sensor = ctypes.c_void_p()
    self.Sdk.GoSdk_Construct(byref(self.api))
    self.Sdk.GoSystem_Construct(byref(self.system), None)
    ip_addr = G.kIpAddress()
    self.kApi.kIpAddress_Parse(byref(ip_addr), ip.encode())
    self.Sdk.GoSystem_FindSensorByIpAddress(self.system, byref(ip_addr), byref(self.sensor))
    self.Sdk.GoSensor_Connect(self.sensor)
    self.Sdk.GoSystem_EnableData(self.system, True)
    self.Sdk.GoSensor_Stop(self.sensor)
    print(f"[+] Gocator connected at {ip}")

  def _surface_points(self, msg):
    Sdk = self.Sdk
    xr = Sdk.GoUniformSurfaceMsg_XResolution(msg) / 1e6     # mm
    yr = Sdk.GoUniformSurfaceMsg_YResolution(msg) / 1e6
    zr = Sdk.GoUniformSurfaceMsg_ZResolution(msg) / 1e6
    xo = Sdk.GoUniformSurfaceMsg_XOffset(msg) / 1e3
    yo = Sdk.GoUniformSurfaceMsg_YOffset(msg) / 1e3
    zo = Sdk.GoUniformSurfaceMsg_ZOffset(msg) / 1e3
    w = Sdk.GoUniformSurfaceMsg_Width(msg)
    n = Sdk.GoUniformSurfaceMsg_Length(msg)
    ptr = Sdk.GoUniformSurfaceMsg_RowAt(msg, 0)
    z = np.ctypeslib.as_array(ptr, shape=(w * n,)).astype(np.float64)
    ok = z != -32768
    self.last_info = dict(width=w, length=n, valid=int(ok.sum()), z_offset_mm=zo,
                          z_res_mm=zr, x_offset_mm=xo, x_res_mm=xr)
    x = np.tile(np.arange(w) * xr + xo, n)
    y = np.repeat(np.arange(n) * yr + yo, w)
    # Z is negated: as read, (x, y, z) is a MIRRORED frame. Found 11/01 from the
    # hand-eye: 11 sphere captures fit to 0.08 mm with z flipped vs 6.7 mm without.
    # With the flip it is a proper right-handed frame (z grows away from the sensor).
    xyz = np.stack([x, y, -(z * zr + zo)], axis=1)[ok]
    return xyz / 1000.0                                       # m

  def snapshot(self, retries=2):
    """One snapshot -> (N, 3) points in the sensor frame, meters."""
    for _ in range(retries + 1):
      self.Sdk.GoSensor_Snapshot(self.sensor)
      t0 = time.time()
      while time.time() - t0 < self.timeout_us / 1e6:
        ds = ctypes.c_void_p()
        if self.Sdk.GoSystem_ReceiveData(self.system, byref(ds), self.timeout_us) != 1:
          break
        pts = None
        for i in range(self.Sdk.GoDataSet_Count(ds)):
          msg = ctypes.c_void_p(self.Sdk.GoDataSet_At(ds, i))
          if self.Sdk.GoDataMsg_Type(msg) == self.G.GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE:
            pts = self._surface_points(msg)
        self.G.kObject_Destroy(ds)
        if pts is not None:
          return pts
      print("[-] Gocator: no surface data, retrying")
    raise RuntimeError("Gocator returned no uniform-surface data (check the job)")

  def close(self):
    self.Sdk.GoSensor_Stop(self.sensor)
    # no kObject_Destroy here: it segfaulted on the Jetson (11/01); the OS frees it at exit


def save_ply(path, pts):
  with open(path, "w") as f:
    f.write("ply\nformat ascii 1.0\n"
            f"element vertex {len(pts)}\nproperty float x\nproperty float y\n"
            "property float z\nend_header\n")
    np.savetxt(f, pts, fmt="%.6f")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--ip", default=GOCATOR_IP)
  ap.add_argument("--n", type=int, default=1)
  ap.add_argument("--out", default="gocator_test.ply")
  a = ap.parse_args()
  g = GocatorCamera(a.ip)
  try:
    for i in range(a.n):
      t = time.time()
      p = g.snapshot()
      inf = g.last_info
      if len(p) == 0:
        print(f"  snapshot {i + 1}: 0 valid points (grid {inf.get('width')} x {inf.get('length')}, "
              f"z offset {inf.get('z_offset_mm')} mm). Everything is out of range: "
              "the Gocator 3210 only measures ~164-274 mm from its front face. "
              "Jog so the object is ~20 cm below the sensor and try again.")
        continue
      print(f"  snapshot {i + 1}: {len(p)} valid of {inf['width'] * inf['length']} "
            f"({inf['width']} x {inf['length']}), {time.time() - t:.2f} s, "
            f"z {1000 * p[:, 2].min():.1f}..{1000 * p[:, 2].max():.1f} mm")
    if len(p) == 0:
      raise SystemExit("[-] no valid points, nothing saved")
    save_ply(a.out, p)
    print(f"[+] last snapshot saved to {a.out} (meters)")
  finally:
    g.close()
