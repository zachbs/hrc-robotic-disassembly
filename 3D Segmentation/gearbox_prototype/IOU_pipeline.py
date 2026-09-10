import copy
from datetime import datetime
import os
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO


def save_processed_scan(pcd, output_dir="captured_scans", filename=None):
  """Saves the cleaned and processed source point cloud to disk.

  Preserves spatial coordinates and computed surface normals.
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[+] Created storage directory: '{output_dir}'")

  if filename is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gearbox_scan_{timestamp}.pcd"

  filepath = os.path.join(output_dir, filename)

  print(f"[*] Saving point cloud to disk...")
  success = o3d.io.write_point_cloud(filepath, pcd)

  if success:
    print(f"[+] Scan safely archived at: {filepath}")
  else:
    print(f"[-] CRITICAL ERROR: Open3D failed to write file to {filepath}")

  return filepath

def mat4x4_to_pose6d(matrix, seq='xyz', degrees=True):
    """
    Converts a 4x4 transformation matrix to [rx, ry, rz, x, y, z].
    
    :param matrix: 4x4 numpy array
    :param seq: Euler angle sequence (e.g., 'xyz' or 'zyx')
    :param degrees: Return angles in degrees if True, else radians
    :return: List of 6 elements [rx, ry, rz, x, y, z]
    """
    # 1. Extract the 3x3 rotation matrix (top-left)
    rot_matrix = matrix[0:3, 0:3]
    
    # 2. Extract the 3x1 translation vector (top-right column)
    translation = matrix[0:3, 3]
    x, y, z = translation[0], translation[1], translation[2]
    
    # 3. Convert the rotation matrix to Euler angles
    rotation = R.from_matrix(rot_matrix)
    rx, ry, rz = rotation.as_euler(seq, degrees=degrees)
    
    # 4. Combine into the final 6D vector
    return [rx, ry, rz, x, y, z]


def compute_2d_iou(boxA, boxB):
  """Computes Intersection over Union (IoU) between two 2D boxes [x1, y1, x2, y2]."""
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB - xA) * max(0, yB - yA)
  if interArea == 0:
    return 0.0

  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def verify_6d_pose_with_yolo(cad_vertices, R, t, K, yolo_bbox, img_shape):
  """Transforms CAD vertices by 6D pose (R, t), projects to 2D image space using intrinsics K, and computes 2D Bounding Box IoU against a YOLO bounding box."""
  # Handle both Open3D PinholeCameraIntrinsic and NumPy 3x3 matrix
  if hasattr(K, "intrinsic_matrix"):
    K_mat = K.intrinsic_matrix
  else:
    K_mat = K

  # 1. Transform CAD vertices to Camera Coordinates
  t = t.reshape(1, 3)
  pts_cam = (R @ cad_vertices.T).T + t

  # Filter out any vertices behind the camera plane (Z <= 0)
  valid_pts = pts_cam[pts_cam[:, 2] > 0]
  if len(valid_pts) == 0:
    return 0.0, [0, 0, 0, 0]

  # 2. Project 3D points to 2D pixel space using camera intrinsics
  fx, fy = K_mat[0, 0], K_mat[1, 1]
  cx, cy = K_mat[0, 2], K_mat[1, 2]

  u = (fx * valid_pts[:, 0] / valid_pts[:, 2]) + cx
  v = (fy * valid_pts[:, 1] / valid_pts[:, 2]) + cy

  # Clamp projected points to image dimensions
  h, w = img_shape[:2]
  u = np.clip(u, 0, w - 1)
  v = np.clip(v, 0, h - 1)

  # 3. Form projected 2D Axis-Aligned Bounding Box
  proj_bbox = [np.min(u), np.min(v), np.max(u), np.max(v)]

  # 4. Calculate IoU against YOLO Bounding Box
  iou = compute_2d_iou(yolo_bbox, proj_bbox)

  return iou, proj_bbox


def visualize_reprojection_result(
    color_img,
    yolo_bbox,
    proj_bbox,
    iou,
    cad_pts=None,
    R=None,
    t=None,
    K=None,
):
  """Overlays YOLO bounding box (Green), CAD projected bounding box (Red), and 3D CAD points projected into 2D camera space (Cyan) on the captured RGB image."""
  display_img = color_img.copy()

  # 1. Draw YOLO Bounding Box in GREEN
  x1, y1, x2, y2 = yolo_bbox
  cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  cv2.putText(
      display_img,
      "YOLO 2D Box",
      (x1, max(y1 - 8, 15)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 255, 0),
      2,
  )

  # 2. Draw Projected CAD Bounding Box in RED
  px1, py1, px2, py2 = map(int, proj_bbox)
  cv2.rectangle(display_img, (px1, py1), (px2, py2), (0, 0, 255), 2)
  cv2.putText(
      display_img,
      f"CAD Projected Box (IoU: {iou:.3f})",
      (px1, max(py1 - 8, 15)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 0, 255),
      2,
  )

  # 3. Project CAD 3D points directly as point overlay (Cyan dots)
  if cad_pts is not None and R is not None and t is not None and K is not None:
    if hasattr(K, "intrinsic_matrix"):
      K_mat = K.intrinsic_matrix
    else:
      K_mat = K

    t_vec = t.reshape(1, 3)
    pts_cam = (R @ cad_pts.T).T + t_vec
    valid = pts_cam[pts_cam[:, 2] > 0]

    if len(valid) > 0:
      fx, fy = K_mat[0, 0], K_mat[1, 1]
      cx, cy = K_mat[0, 2], K_mat[1, 2]
      u = ((fx * valid[:, 0] / valid[:, 2]) + cx).astype(int)
      v = ((fy * valid[:, 1] / valid[:, 2]) + cy).astype(int)

      h_img, w_img = display_img.shape[:2]
      # Downsample point rendering for speed
      step = max(1, len(u) // 1200)
      for px, py in zip(u[::step], v[::step]):
        if 0 <= px < w_img and 0 <= py < h_img:
          cv2.circle(display_img, (px, py), 1, (255, 255, 0), -1)

  # 4. Display Verification Window
  window_title = f"6D Reprojection Verification - IoU: {iou:.4f}"
  cv2.imshow(window_title, display_img)
  print(f"\n[+] Showing 2D Reprojection Overlay window. Press ANY KEY to continue...")
  cv2.waitKey(0)
  cv2.destroyWindow(window_title)


# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved-Fusion.stl"
PCD_FILE_PATH = "captured_scans/origin_centered_local_global.pcd"
YOLO_MODEL_PATH = "06-09-2026.pt"
VOXEL_SIZE = 0.0025
PADDING = 20  # YOLO bounding box 2D padding
FRAMES_TO_CAPTURE = 25


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def draw_registration_step(source, target, transformation, window_name):
  """Helper to visualize alignment steps with consistent coloring."""
  source_temp = copy.deepcopy(source)
  target_temp = copy.deepcopy(target)

  source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow = Scan
  target_temp.paint_uniform_color([0, 0.651, 0.929])  # Cyan = CAD Model

  source_temp.transform(transformation)
  o3d.visualization.draw_geometries(
      [source_temp, target_temp], window_name=window_name
  )


def extract_fpfh_features(pcd, voxel_size, is_source=False):
  """Computes geometric surface normals consistently and extracts FPFH descriptors."""
  if is_source:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )
  else:
    if not pcd.has_normals():
      print(
          "[*] CAD Target missing normals. Computing dynamically scaled"
          " vectors..."
      )
      pcd.estimate_normals(
          o3d.geometry.KDTreeSearchParamHybrid(
              radius=voxel_size * 2, max_nn=30
          )
      )

  o3d.visualization.draw_geometries(
      [pcd],
      window_name="FPFH Normal Verification",
      point_show_normal=True,
  )
  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd,
      o3d.geometry.KDTreeSearchParamHybrid(
          radius=radius_feature, max_nn=100
      ),
  )
  return fpfh


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
  print("\n[*] Initializing YOLO & RealSense...")
  yolo_model = YOLO(YOLO_MODEL_PATH)

  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

  # Start streaming pipeline
  pipeline_profile = pipeline.start(config)

  # --------------------------------------------------------------------------
  # DYNAMIC INTRINSICS EXTRACTION (Works with any RealSense model)
  # --------------------------------------------------------------------------
  align = rs.align(rs.stream.color)
  color_stream_profile = pipeline_profile.get_stream(
      rs.stream.color
  ).as_video_stream_profile()
  intr = color_stream_profile.get_intrinsics()

  INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
      width=intr.width,
      height=intr.height,
      fx=intr.fx,
      fy=intr.fy,
      cx=intr.ppx,
      cy=intr.ppy,
  )
  print(
      f"[+] Dynamic Camera Intrinsics Loaded: {intr.width}x{intr.height},"
      f" fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f},"
      f" cy={intr.ppy:.2f}"
  )
  time.sleep(1)
  pipeline_profile = pipeline.get_active_profile()
  device = pipeline_profile.get_device()
  depth_sensor = device.first_depth_sensor()
  
  
  # 1. Maximize IR Laser Power (Default range is usually 0 to 360 mW)
  if depth_sensor.supports(rs.option.laser_power):
      depth_sensor.set_option(
      rs.option.laser_power, 360.0
      )  # Set to max laser power
  
  # 2. Disable Auto Exposure and set manual higher exposure
  if depth_sensor.supports(rs.option.enable_auto_exposure):
      depth_sensor.set_option(rs.option.enable_auto_exposure, False)
  
  if depth_sensor.supports(rs.option.exposure):
      # Exposure value in microseconds (e.g., 16000us = 16ms)
      depth_sensor.set_option(rs.option.exposure, 16000.0)
  
  print("RealSense IR Emitter and Exposure configured successfully!")
  time.sleep(1)

  print("[*] Preparing CAD Target...")
  pristine_target = o3d.io.read_point_cloud(PCD_FILE_PATH)

  # --------------------------------------------------------------------------
  # DATA GATHERING LOOP
  # --------------------------------------------------------------------------
  last_captured_color_frame = None
  try:
    locked_box = None
    depth_buffer = []

    print("\n[*] Streaming live preview. Press 's' to capture burst...")
    while True:
      frames = pipeline.wait_for_frames()
      aligned_frames = align.process(frames)
      depth_frame = aligned_frames.get_depth_frame()
      color_frame = aligned_frames.get_color_frame()

      if not depth_frame or not color_frame:
        continue

      depth_image = np.asanyarray(depth_frame.get_data()).copy()
      color_image = np.asanyarray(color_frame.get_data()).copy()

      results = yolo_model(color_image, conf=0.60, verbose=False)[0]
      preview = results.plot()
      cv2.imshow("Live Stream (Press 's' to Trigger Burst)", preview)

      key = cv2.waitKey(1) & 0xFF

      if key == ord("s") or key == ord("S"):
        if len(results.boxes) > 0:
          valid_box = tuple(map(int, results.boxes[0].xyxy[0]))
          print(
              f"\n[+] 's' pressed! Target spotted. Capturing"
              f" {FRAMES_TO_CAPTURE} frames..."
          )

          # Store the exact color image from the trigger moment
          last_captured_color_frame = color_image.copy()

          depth_buffer = [depth_image]
          while len(depth_buffer) < FRAMES_TO_CAPTURE:
            b_frames = pipeline.wait_for_frames()
            b_aligned = align.process(b_frames)
            b_depth = b_aligned.get_depth_frame()
            if b_depth:
              depth_buffer.append(np.asanyarray(b_depth.get_data()).copy())

          locked_box = valid_box
          cv2.destroyAllWindows()
          print("[+] Target Locked. Burst capture successful.")
          break
        else:
          print(
              "[-] CAPTURE REJECTED: YOLO does not detect the gearbox in this"
              " frame view!"
          )

      elif key == ord("q") or key == 27:
        print("[*] User aborted tracking pipeline.")
        return

  finally:
    pipeline.stop()
    cv2.destroyAllWindows()

  # --------------------------------------------------------------------------
  # STEP 1: TEMPORAL MEDIAN
  # --------------------------------------------------------------------------
  print("\n[STEP 1] Computing Temporal Median...")
  depth_stack = np.stack(depth_buffer, axis=0)
  median_depth = np.median(depth_stack, axis=0).astype(np.uint16)

  # --------------------------------------------------------------------------
  # STEP 2: SPATIAL BILATERAL FILTERING
  # --------------------------------------------------------------------------
  print("[STEP 2] Applying Edge-Preserving Bilateral Filter...")
  depth_f32 = median_depth.astype(np.float32)
  filtered_depth_f32 = cv2.bilateralFilter(
      depth_f32, d=5, sigmaColor=15.0, sigmaSpace=5.0
  )
  filtered_depth = filtered_depth_f32.astype(np.uint16)

  # --------------------------------------------------------------------------
  # STEP 3: YOLO ROI MASKING (2D Frustum Isolation)
  # --------------------------------------------------------------------------
  print("[STEP 3] Cropping ROI based on YOLO bounding box...")
  x1, y1, x2, y2 = locked_box
  h, w = filtered_depth.shape
  y1_pad, y2_pad = max(0, y1 - PADDING), min(h, y2 + PADDING)
  x1_pad, x2_pad = max(0, x1 - PADDING), min(w, x2 + PADDING)

  masked_depth = np.zeros_like(filtered_depth)
  roi = filtered_depth[y1_pad:y2_pad, x1_pad:x2_pad]
  depth_mask = (roi > 0.0) & (roi <= 800.0)
  masked_depth[y1_pad:y2_pad, x1_pad:x2_pad] = np.where(depth_mask, roi, 0.0)

  # --------------------------------------------------------------------------
  # STEP 4: 3D POINT CLOUD PROJECTION
  # --------------------------------------------------------------------------
  print("[STEP 4] Generating 3D Point Cloud...")
  depth_img = o3d.geometry.Image(masked_depth)
  pcd = o3d.geometry.PointCloud.create_from_depth_image(
      depth=depth_img,
      intrinsic=INTRINSICS,
      depth_scale=1000.0,
      depth_trunc=0.8,
  )

  print(f"-> Raw projected point cloud contains {len(pcd.points)} points.")
  pcd.paint_uniform_color([0.5, 0.5, 0.5])

  # --------------------------------------------------------------------------
  # STEP 5: VOXEL DOWNSAMPLING
  # --------------------------------------------------------------------------
  print("\n[STEP 5] Voxel Downsampling...")
  pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
  print(f"-> Downsampled to {len(pcd.points)} points.")

  # --------------------------------------------------------------------------
  # STEP 7: OUTLIER REMOVAL
  # --------------------------------------------------------------------------
  print("\n[STEP 7] Removing Statistical & Radius Outliers...")
  pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
  print(f"-> Clean target object isolated with {len(pcd.points)} points.")

  source = pcd
  source.paint_uniform_color([1, 0.706, 0])  # Yellow

  # --------------------------------------------------------------------------
  # STEP 8: CAMERA-ORIENTED NORMAL ESTIMATION
  # --------------------------------------------------------------------------
  print("\n[STEP 8] Estimating and Orienting Surface Normals...")
  source.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30)
  )
  source.orient_normals_towards_camera_location(
      camera_location=np.array([0.0, 0.0, 0.0])
  )

  # --------------------------------------------------------------------------
  # STEP 9: FPFH DESCRIPTOR EXTRACTION
  # --------------------------------------------------------------------------
  print("\n[STEP 9] Computing FPFH Descriptors...")
  source_fpfh = extract_fpfh_features(source, VOXEL_SIZE, is_source=True)
  target_fpfh = extract_fpfh_features(
      pristine_target, VOXEL_SIZE, is_source=False
  )

  # --------------------------------------------------------------------------
  # STEP 10: RANSAC GLOBAL COARSE ALIGNMENT
  # --------------------------------------------------------------------------
  print("\n[STEP 10] Running RANSAC Global Alignment...")
  translation_vec = source.get_center() - pristine_target.get_center()
  T_init = np.eye(4)
  T_init[:3, 3] = translation_vec
  target_copy = copy.deepcopy(pristine_target)

  target_copy.translate(translation_vec)
  target_copy.paint_uniform_color([0, 0.651, 0.929])
  distance_threshold = VOXEL_SIZE * 1.5
  normal_cos_threshold = np.cos(np.radians(45.0))
  start_time = time.time()

  ransac_result = (
      o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
          source,
          target_copy,
          source_fpfh,
          target_fpfh,
          mutual_filter=True,
          max_correspondence_distance=distance_threshold,
          estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
          ransac_n=3,
          checkers=[
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                  0.88
              ),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                  distance_threshold
              ),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                  normal_cos_threshold
              ),
          ],
          criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
              1000000, 1000
          ),
      )
  )

  print(
      f"-> RANSAC completed in {time.time() - start_time:.2f}s (Fitness:"
      f" {ransac_result.fitness:.4f}) | Inlier RMSE:"
      f" {ransac_result.inlier_rmse:.4f}"
  )
  draw_registration_step(
      source,
      target_copy,
      ransac_result.transformation,
      "Step 10: RANSAC Coarse Alignment",
  )

  # --------------------------------------------------------------------------
  # STEP 11: MULTI-SCALE POINT-TO-PLANE ICP REFINEMENT
  # --------------------------------------------------------------------------
  print("\n[STEP 11] Running Multi-Scale Local ICP...")
  icp_thresholds = [0.012, 0.010, 0.008]
  T_current = ransac_result.transformation
  start_time = time.time()

  for i, threshold in enumerate(icp_thresholds):
    print(f"   -> Scale Stage {i+1} (Search Radius: {threshold*1000:.1f}mm)...")
    icp_result = o3d.pipelines.registration.registration_icp(
        source,
        target_copy,
        threshold,
        init=T_current,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100
        ),
    )
    T_current = icp_result.transformation

  print(
      f"-> ICP completed in {time.time() - start_time:.2f}s (Fitness:"
      f" {icp_result.fitness:.4f}) | Inlier RMSE: {icp_result.inlier_rmse:.4f}"
  )
  # 1. Invert the Camera -> CAD registration matrix
  M_inv = np.linalg.inv(T_current)

  # 2. Compose with T_init: Maps pristine CAD -> Camera space
  T_cad_to_cam = M_inv @ T_init

  # Extract rotation and translation directly
  R_final = T_cad_to_cam[:3, :3]
  t_final = T_cad_to_cam[:3, 3]
  print("\n[+] Final Optimized 6D Pose Transformation Matrix:\n", T_cad_to_cam)
  print("\n[+] Final Optimized 6D Pose (Euler Angles + Translation):\n", mat4x4_to_pose6d(T_cad_to_cam))
  draw_registration_step(
      pristine_target, source, T_cad_to_cam, "Step 11: Multi-Scale ICP Refinement"
  )

  # --------------------------------------------------------------------------
  # REPROJECTION VERIFICATION & OVERLAY VISUALIZATION
  # --------------------------------------------------------------------------
  # Invert the Camera -> CAD matrix to get CAD -> Camera
 
  iou, proj_bbox = verify_6d_pose_with_yolo(
      cad_vertices=np.asarray(pristine_target.points),
      R=R_final,
      t=t_final,
      K=INTRINSICS,
      yolo_bbox=locked_box,
      img_shape=(480, 640),
  )

  print(f"\n[+] YOLO Bounding Box: {locked_box}")
  print(f"[+] Projected CAD Bounding Box: {proj_bbox}")
  print(f"[+] 2D Bounding Box IoU: {iou:.4f}")

  # Display the Captured Frame with Reprojected Overlays
  visualize_reprojection_result(
      color_img=last_captured_color_frame,
      yolo_bbox=locked_box,
      proj_bbox=proj_bbox,
      iou=iou,
      cad_pts=np.asarray(pristine_target.points),
      R=R_final,
      t=t_final,
      K=INTRINSICS,
  )


if __name__ == "__main__":
  main()