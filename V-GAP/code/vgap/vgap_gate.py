"""V-GAP consistency gate helpers for the gearbox (and the other three objects).

Drop-in functions for IOU_pipeline.py. Nothing here talks to the camera or YOLO
directly, so it can be tested offline. Units are meters, poses are 4x4
CAD->camera matrices (the T_cad_to_cam your pipeline already computes).

Plan equations implemented:
  IB, IM, C            Eq. 5-7   (bounding-box IoU, mask IoU, weighted score)
  Valid / Conflict     Eq. 8
  UD, Ulive            Eq. 9-10
  symmetric rot. error Eq. 18
Gearbox addition: resolve_yaw() tests the 90 deg yaw alternative, which is the
failure mode when the fastener lips have no depth.
"""

import copy

import cv2
import numpy as np
import open3d as o3d


# ------------------------------------------------------------------------------
# Depth preprocessing
# ------------------------------------------------------------------------------
def temporal_median_ignore_zeros(depth_buffer, min_valid_frac=0.3):
  """Per-pixel median over the burst, ignoring invalid (0) readings.

  np.median over raw frames counts 0 as a value, so any pixel that is invalid
  in more than half the frames becomes 0. On dark or reflective filament the
  thin lips are often valid only in some frames; this keeps them.
  """
  stack = np.stack(depth_buffer, axis=0).astype(np.float32)
  stack[stack == 0] = np.nan
  valid_frac = np.mean(~np.isnan(stack), axis=0)
  with np.errstate(all="ignore"):
    med = np.nanmedian(stack, axis=0)
  med[(valid_frac < min_valid_frac) | np.isnan(med)] = 0
  return med.astype(np.uint16)


def yolo_mask(result, img_shape, det_index=0):
  """Returns (M2D bool HxW, B2D [x1,y1,x2,y2]) from an ultralytics seg result."""
  h, w = img_shape[:2]
  box = result.boxes[det_index].xyxy[0].cpu().numpy()
  if result.masks is None:
    raise ValueError("YOLO result has no masks: load the -seg model.")
  m = result.masks.data[det_index].cpu().numpy()
  m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
  return m, [int(v) for v in box]


def masked_depth(depth, mask, erode_px=2):
  """Keeps depth only inside the (slightly eroded) object mask.

  Replaces the padded-bbox crop: no table or background pixels reach
  RANSAC, so the table-plane removal step is no longer needed.
  """
  if erode_px > 0:
    k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), k) > 0
  return np.where(mask, depth, 0).astype(np.uint16)


def depth_incompleteness(depth, mask):
  """UD = 1 - Nvalid / Nmask  (Eq. 9)."""
  n_mask = int(mask.sum())
  if n_mask == 0:
    return 1.0
  return 1.0 - float(((depth > 0) & mask).sum()) / n_mask


# ------------------------------------------------------------------------------
# CAD projection and consistency
# ------------------------------------------------------------------------------
def project_silhouette(cad_points, T_cad_to_cam, K, img_shape, close_px=3):
  """Mproj and Bproj (Eq. 3-4): splats dense CAD points and closes the gaps.

  cad_points should be dense (e.g. 20-50k points sampled from the exterior
  mesh) so the splat is solid at the working distance.
  """
  h, w = img_shape[:2]
  pts = (T_cad_to_cam[:3, :3] @ cad_points.T).T + T_cad_to_cam[:3, 3]
  pts = pts[pts[:, 2] > 1e-6]
  u = np.round(K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]).astype(int)
  v = np.round(K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]).astype(int)
  mask = np.zeros((h, w), np.uint8)
  ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
  mask[v[ok], u[ok]] = 1
  if close_px > 0:
    k = np.ones((2 * close_px + 1, 2 * close_px + 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
  mask = mask > 0
  if not mask.any():
    return mask, [0, 0, 0, 0]
  ys, xs = np.nonzero(mask)
  return mask, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_iou(a, b):
  ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
  iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
  inter = ix * iy
  union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return inter / union if union > 0 else 0.0


def mask_iou(a, b):
  union = np.logical_or(a, b).sum()
  return float(np.logical_and(a, b).sum()) / union if union > 0 else 0.0


def consistency(M2D, B2D, Mproj, Bproj, wB=0.3, wM=0.7):
  """Returns (IB, IM, C) per Eq. 5-7."""
  IB = box_iou(B2D, Bproj)
  IM = mask_iou(M2D, Mproj)
  return IB, IM, wB * IB + wM * IM


def gate(C, tau_c):
  """Eq. 8."""
  return "Valid" if C >= tau_c else "Conflict"


def live_uncertainty(UD, C, alpha=0.7):
  """Eq. 10."""
  return alpha * UD + (1 - alpha) * (1 - C)


# ------------------------------------------------------------------------------
# Gearbox symmetry
# ------------------------------------------------------------------------------
def principal_frame(cad_points, snap_deg=10.0):
  """Center and axes of the CAD cloud: columns = long, short, height axis.

  Axes come from PCA and are snapped to the CAD's own x/y/z when within
  `snap_deg` (CAD exports are normally axis-aligned). Snapping matters: a
  few degrees of PCA tilt turns the "harmless" 180 deg flip into a real
  rotation error. Use the exterior-only CAD cloud; interior walls skew PCA.
  The center is the bounding-box center, which is exact for a symmetric part.
  """
  c = 0.5 * (cad_points.min(axis=0) + cad_points.max(axis=0))
  _, _, vt = np.linalg.svd(cad_points - cad_points.mean(axis=0),
                           full_matrices=False)
  axes = vt.T  # sorted by decreasing variance
  for i in range(3):
    j = int(np.argmax(np.abs(axes[:, i])))
    if np.abs(axes[j, i]) >= np.cos(np.radians(snap_deg)):
      axes[:, i] = np.sign(axes[j, i]) * np.eye(3)[:, j]
  if np.linalg.det(axes) < 0:
    axes[:, 2] *= -1
  return c, axes


def rotation_about(center, axis, angle_deg):
  """4x4 rotation about an axis through `center`, in the CAD frame."""
  rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
      np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(angle_deg))
  T = np.eye(4)
  T[:3, :3] = rot
  T[:3, 3] = center - rot @ center
  return T


def gearbox_symmetry_set(cad_points):
  """D2: identity and 180 deg about each principal axis, as 3x3 rotations
  expressed in the CAD frame (for Eq. 18)."""
  _, axes = principal_frame(cad_points)
  G = [np.eye(3)]
  for i in range(3):
    G.append(o3d.geometry.get_rotation_matrix_from_axis_angle(axes[:, i] * np.pi))
  return G


def symmetric_rotation_error_deg(R_gt, R_est, G):
  """Eq. 18: min over S in G of the angle between R_gt S and R_est."""
  best = 180.0
  for S in G:
    c = (np.trace((R_gt @ S).T @ R_est) - 1) / 2
    best = min(best, np.degrees(np.arccos(np.clip(c, -1, 1))))
  return best


def refine(cad_pcd, source, T, thresholds=(0.008, 0.005, 0.003)):
  """Multi-scale point-to-plane ICP, CAD -> scan. Returns (T, fitness)."""
  fit = 0.0
  for th in thresholds:
    r = o3d.pipelines.registration.registration_icp(
        cad_pcd, source, th, init=T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T, fit = r.transformation, r.fitness
  return T, fit


def resolve_yaw(source, cad_pcd, T_cad_to_cam, K, M2D, B2D, img_shape,
                cad_dense_points=None, wB=0.3, wM=0.7):
  """Tests the pose and its 90 deg yaw alternative; keeps the one the 2D
  evidence prefers.

  For the gearbox the 180 deg flips are harmless (D2 symmetry), so only
  0 and 90 deg about the height axis need comparing. Each candidate gets a
  short point-to-plane ICP (CAD -> scan) and is scored with C.

  Returns dict(T, C, IB, IM, margin, chose_rotated, scores).
  margin = C(best) - C(other); small margin means the short axis is
  ambiguous from this view.
  """
  pts = np.asarray(cad_pcd.points)
  dense = pts if cad_dense_points is None else cad_dense_points
  center, axes = principal_frame(pts)
  if not source.has_normals():
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.006, max_nn=30))

  scores = []
  for yaw in (0.0, 90.0):
    T0 = T_cad_to_cam @ rotation_about(center, axes[:, 2], yaw)
    T1, fit = refine(cad_pcd, source, T0)
    Mp, Bp = project_silhouette(dense, T1, K, img_shape)
    IB, IM, C = consistency(M2D, B2D, Mp, Bp, wB, wM)
    scores.append(dict(yaw=yaw, T=T1, IB=IB, IM=IM, C=C, fitness=fit))

  best, other = sorted(scores, key=lambda s: s["C"], reverse=True)
  return dict(T=best["T"], C=best["C"], IB=best["IB"], IM=best["IM"],
              margin=best["C"] - other["C"], chose_rotated=best["yaw"] != 0.0,
              scores=scores)


def gearbox_decision(C, margin, tau_c, delta):
  """Decision table from the notes (Section 3)."""
  if C < tau_c:
    return "Conflict", "geometry"
  if margin < delta:
    return "Conflict", "short-axis ambiguous"
  return "Valid", ""
