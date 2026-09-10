import profile

import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import time
import os

from datetime import datetime

from torch import device
from ultralytics import YOLO


PCD_FILE_PATH = "captured_scans/origin_centered_local_global.pcd"
NP_SCAN_PATH = "captured_scans/feature_mapping.npy"
BBOX_GB_PATH = "captured_scans/gearbox_bbox.npy"
BBOX_FASTNER1_PATH = "captured_scans/fastener_1.npy"
BBOX_FASTNER2_PATH = "captured_scans/fastener_2.npy"
YOLO_MODEL_PATH = "06-09-2026.pt"
PADDING = 20  # YOLO bounding box 2D padding
FRAMES_TO_CAPTURE = 25
VOXEL_SIZE = 0.0025
take_snapshot = False  # Set to False to load pre-captured data from disk

# INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
#         width=640, 
#         height=480, 
#         fx=609.3075561523438,  # Corrected from 425.0
#         fy=608.9049072265625,  # Corrected from 425.0  # Focal length Y
#         cx=326.06304931640625,  # Principal point X
#         cy=249.21212768554688   # Principal point Y
#     )
# import open3d as o3d

# Updated Open3D intrinsic object matching your new RealSense settings
INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=381.213196,
    fy=381.213196,
    cx=324.086700,
    cy=240.891968,
)



def save_processed_scan(np_array, output_dir="captured_scans", filename=None):
    """
    Saves the cleaned and processed source point cloud to disk.
    Preserves spatial coordinates and computed surface normals.
    """
    if np_array is None or not isinstance(np_array, np.ndarray):
        print("[-] ERROR: Invalid input array. Cannot save.")
        return None
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[+] Created storage directory: '{output_dir}'")
        
    # Generate a unique timestamped filename if none is provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gearbox_scan_{timestamp}.pcd"
        
    filepath = os.path.join(output_dir, filename)
    
    try:
        np.save(filepath, np_array)
        print(f"[+] Scan safely archived at: {filepath}")
        return filepath
    except Exception as e:
        print(f"[-] CRITICAL ERROR: Failed to write file to {filepath}. Error: {e}")
        return None
        




def main():
    print("\n[*] Initializing YOLO & RealSense...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
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
    
# --------------------------------------------------------------------------
    # DATA GATHERING LOOP
    # --------------------------------------------------------------------------
    try:
        locked_box = None
        depth_buffer = []
        fastners_detected = []
        
        print("\n[*] Streaming live preview. Press 's' to capture burst...")
        while True:
            if not take_snapshot:
                break  # Exit the loop if a snapshot has been taken
            # 1. Grab continuous live preview frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()
            
            # 2. Run continuous YOLO inference for the live GUI preview
            results = yolo_model(color_image, conf=0.60, verbose=False)[0]
            preview = results.plot()
            cv2.imshow("Live Stream (Press 's' to Trigger Burst)", preview)
            
            # 3. Check for key press every frame (1ms delay keeps stream fluid)
            key = cv2.waitKey(1) & 0xFF
            
            # 4. If user presses 's', validate and initiate the burst capture
            if key == ord('s') or key == ord('S'):
                if len(results.boxes) >= 3:
                    valid_box = tuple(map(int, results.boxes[0].xyxy[0]))
                    save_processed_scan(results.boxes[0].xyxy[0].cpu().numpy(), output_dir="captured_scans", filename="gearbox_bbox.npy")
                    num_fastners = 0
                    
                    for box in results.boxes[1:]:
                        if float(box.conf[0]) >= 0.80 and int(box.cls[0]) == 1: # Assuming class is fastner 
                            num_fastners += 1
                            fastners_detected.append(tuple(map(int, box.xyxy[0])))
                            save_processed_scan(box.xyxy[0].cpu().numpy(), output_dir="captured_scans", filename=f"fastener_{num_fastners}.npy")
                    if num_fastners < 2:
                        print("Not enough fasteners")
                        continue  # Skip this frame if not enough fasteners detected

                    print(f"\n[+] 's' pressed! Target spotted. Capturing {FRAMES_TO_CAPTURE} frames...")
                    
                    # Initialize buffer with the current validated frame
                    depth_buffer = [depth_image]
                    
                    # Fast-capture loop for the remaining frames in the burst
                    while len(depth_buffer) < FRAMES_TO_CAPTURE:
                        b_frames = pipeline.wait_for_frames()
                        b_aligned = align.process(b_frames)
                        b_depth = b_aligned.get_depth_frame()
                        if b_depth:
                            depth_buffer.append(np.asanyarray(b_depth.get_data()).copy())
                    
                    # Secure the lock and cleanly break out of the streaming loop
                    locked_box = valid_box
                    cv2.destroyAllWindows()
                    print("[+] Target Locked. Burst capture successful.")
                    break
                else:
                    print("[-] CAPTURE REJECTED: YOLO does not detect the gearbox in this frame view!")
            
            # Allow clean exit with 'q' or ESC
            elif key == ord('q') or key == 27:
                print("[*] User aborted tracking pipeline.")
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    filtered_depth = None

    if not take_snapshot:
        print("[-] No snapshot was taken. Loading pre-captured data from disk...")
        filtered_depth = np.load(NP_SCAN_PATH)
        locked_box = tuple(map(int, np.load(BBOX_GB_PATH)))
        fastners_detected = [tuple(map(int, np.load(BBOX_FASTNER1_PATH))), tuple(map(int, np.load(BBOX_FASTNER2_PATH)))]
    else:
        print("[+] Snapshot taken. Proceeding with captured data...")
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
        filtered_depth_f32 = cv2.bilateralFilter(depth_f32, d=5, sigmaColor=15.0, sigmaSpace=5.0)
        filtered_depth = filtered_depth_f32.astype(np.uint16)
        save_processed_scan(filtered_depth, output_dir="captured_scans", filename="feature_mapping.npy")



    # --------------------------------------------------------------------------
    # STEP 3: YOLO ROI MASKING (2D Frustum Isolation)
    # --------------------------------------------------------------------------
    
    fastners_depth_masks = []
    median_fastner_depths = []
    h, w = filtered_depth.shape
    #cropping the fasteners detected
    for fastener_box in fastners_detected:
        fx1, fy1, fx2, fy2 = fastener_box
        fy1_pad, fy2_pad = max(0, fy1), min(h, fy2)
        fx1_pad, fx2_pad = max(0, fx1), min(w, fx2)
    
        fastener_roi = filtered_depth[fy1_pad:fy2_pad, fx1_pad:fx2_pad]
        fastener_mask = (fastener_roi > 0.0) & (fastener_roi <= 600.0)
        masked_fastener_depth = np.zeros_like(filtered_depth)
        masked_fastener_depth[fy1_pad:fy2_pad, fx1_pad:fx2_pad] = np.where(fastener_mask, fastener_roi, 0.0)
        valid_depth_values = fastener_roi[fastener_mask]
        if valid_depth_values.size > 0:
            median_fastner_depths.append(np.median(valid_depth_values))
        else:
            median_fastner_depths.append(0.0) 
              # Default to 0 if no valid depth values
        print(f"-> Fastener median depth: {median_fastner_depths[-1]}")
    
        # Populate the masked depth map for fasteners
        fastners_depth_masks.append(masked_fastener_depth)
    
    print("[STEP 3] Cropping ROI based on YOLO bounding box...")
    x1, y1, x2, y2 = locked_box
    y1_pad, y2_pad = max(0, y1 - PADDING), min(h, y2 + PADDING)
    x1_pad, x2_pad = max(0, x1 - PADDING), min(w, x2 + PADDING)

    masked_depth = np.zeros_like(filtered_depth)
    roi = filtered_depth[y1_pad:y2_pad, x1_pad:x2_pad]
    depth_mask = (roi > 0.0) & (roi <= 600.0)

    # Populate the masked depth map
    masked_depth[y1_pad:y2_pad, x1_pad:x2_pad] = np.where(depth_mask, roi, 0.0)


    
    # --------------------------------------------------------------------------
    # STEP 4: 3D POINT CLOUD PROJECTION
    # --------------------------------------------------------------------------
    print("[STEP 4] Generating 3D Point Cloud...")
    depth_img = o3d.geometry.Image(masked_depth)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_img, intrinsic=INTRINSICS, depth_scale=1000.0, depth_trunc=0.8)
    
    print(f"-> Raw projected point cloud contains {len(pcd.points)} points.")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd], window_name="Step 4: Raw 3D ROI Point Cloud")

    fastner_pcds = []
    fastner_median_pcds = []

    for i, fastener_mask in enumerate(fastners_depth_masks):
        fastener_depth_img = o3d.geometry.Image(fastener_mask)
        fastener_pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth=fastener_depth_img, intrinsic=INTRINSICS, depth_scale=1000.0, depth_trunc=0.8)
        fastener_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for fasteners
        fastner_pcds.append(fastener_pcd)
        median_mask = (fastener_mask <= int(median_fastner_depths[i] + 1)) & (fastener_mask >= int(median_fastner_depths[i] - 1))
        median_array = np.zeros_like(fastener_mask, dtype=np.uint16)
        median_array[median_mask] = median_fastner_depths[i]
        fastner_median_depth = o3d.geometry.Image(median_array)
        fastner_pcd_median = o3d.geometry.PointCloud.create_from_depth_image(
            depth=fastner_median_depth, intrinsic=INTRINSICS, depth_scale=1000.0, depth_trunc=0.8)
        fastner_pcd_median.paint_uniform_color([0.0, 1.0, 0.0])  # Green color for median depth
        fastner_median_pcds.append(fastner_pcd_median)

    for i, fastener_pcd in enumerate(fastner_pcds):
        print(f"-> Fastener {i+1} point cloud contains {len(fastener_pcd.points)} points.")
        o3d.visualization.draw_geometries([fastener_pcd, fastner_median_pcds[i]], window_name=f"Fastener {i+1} Point Cloud")

    

    # --------------------------------------------------------------------------
    # STEP 5: VOXEL DOWNSAMPLING
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Voxel Downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"-> Downsampled to {len(pcd.points)} points.")
    o3d.visualization.draw_geometries([pcd], window_name="Step 5: Voxel Downsampled Cloud")

    # for i, fastener_pcd in enumerate(fastner_pcds):
    #     fastener_pcd = fastener_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE/5)
    #     print(f"-> Fastener {i+1} downsampled to {len(fastener_pcd.points)} points.")
    #     o3d.visualization.draw_geometries([fastener_pcd], window_name=f"Fastener {i+1} Voxel Downsampled Cloud")

if __name__ == "__main__":
    main()

  