import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import time
import copy
from ultralytics import YOLO
import os
from datetime import datetime

def save_processed_scan(pcd, output_dir="captured_scans", filename=None):
    """
    Saves the cleaned and processed source point cloud to disk.
    Preserves spatial coordinates and computed surface normals.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[+] Created storage directory: '{output_dir}'")
        
    # Generate a unique timestamped filename if none is provided
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

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved-Fusion.stl"
YOLO_MODEL_PATH = "06-09-2026.pt"
VOXEL_SIZE = 0.001
PADDING = 20  # YOLO bounding box 2D padding
FRAMES_TO_CAPTURE = 25

# RealSense D435 Default Intrinsics
INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
        width=640, 
        height=480, 
        fx=609.3075561523438,  # Corrected from 425.0
        fy=608.9049072265625,  # Corrected from 425.0  # Focal length Y
        cx=326.06304931640625,  # Principal point X
        cy=249.21212768554688   # Principal point Y
    )

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def draw_registration_step(source, target, transformation, window_name):
    """Helper to visualize alignment steps with consistent coloring."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([1, 0.706, 0])      # Yellow = Scan
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Cyan = CAD Model
    
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def extract_fpfh_features(pcd, voxel_size, is_source=False):
    """Computes geometric surface normals consistently and extracts FPFH descriptors."""
    # 1. Only estimate normals if they aren't already present (preserves pristine CAD mesh normals)
    if not pcd.has_normals():
        if is_source:
            radius_normal = 0.004  
        else:
            radius_normal = 0.002
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 2. CRITICAL: Force all normal arrows on the visible surface to point directly at the camera lens
    # Since your RealSense camera is the origin of the coordinate system, the lens position is [0, 0, 0]
    if is_source:
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    else:
        pcd.orient_normals_consistent_tangent_plane(k=15)
    # 3. Compute FPFH now that vectors are perfectly aligned
    radius_feature = voxel_size * 12
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
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
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    time.sleep(1)
    

    print("[*] Preparing CAD Target...")
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    mesh.compute_vertex_normals()  # Generates uniform outward-facing vectors
    pristine_target = mesh.sample_points_uniformly(number_of_points=30000)
    pristine_target.scale(0.00168095, center=pristine_target.get_center())
   
    pristine_target = pristine_target.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    # --------------------------------------------------------------------------
    # DATA GATHERING LOOP
    # --------------------------------------------------------------------------
    try:
        locked_box = None
        depth_buffer = []
        
        print("\n[*] Streaming live preview. Press 's' to capture burst...")
        while True:
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
                if len(results.boxes) > 0:
                    valid_box = tuple(map(int, results.boxes[0].xyxy[0]))
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

    # --------------------------------------------------------------------------
    # STEP 5: VOXEL DOWNSAMPLING
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Voxel Downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"-> Downsampled to {len(pcd.points)} points.")
    o3d.visualization.draw_geometries([pcd], window_name="Step 5: Voxel Downsampled Cloud")

    # --------------------------------------------------------------------------
    # STEP 6: TABLE PLANE SEGMENTATION
    # --------------------------------------------------------------------------
    print("\n[STEP 6] Removing Dominant Table Plane...")
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=200)
        pcd = pcd.select_by_index(inliers, invert=True)
        print(f"-> After table removal, cloud has {len(pcd.points)} points.")
        o3d.visualization.draw_geometries([pcd], window_name="Step 6: Table Plane Removed")
    except Exception as e:
        print(f"-> Plane segmentation failed: {e}")

    # --------------------------------------------------------------------------
    # STEP 7: STATISTICAL & RADIUS OUTLIER REMOVAL (Flying Pixel Cleanup)
    # --------------------------------------------------------------------------
    print("\n[STEP 7] Removing Statistical & Radius Outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.0)
    # # Pass 1: Remove "Deep Space" outliers (Very far, very sparse noise)
    # # This clears out things floating far from the gearbox without touching the model
    # pcd, _ = pcd.remove_radius_outlier(nb_points=15, radius=0.03) 

    # Pass 2: Remove "Near Surface" noise (The edge bleed)
    # Use a tighter radius but a MUCH lower neighbor count to protect the actual surface
    # pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.005)
    print(f"-> Clean target object isolated with {len(pcd.points)} points.")
    
    source = pcd
    source.paint_uniform_color([1, 0.706, 0]) # Yellow
    o3d.visualization.draw_geometries([source], window_name="Step 7: Cleaned Object (Post-Outlier Removal)")

    # --------------------------------------------------------------------------
    # STEP 8: CAMERA-ORIENTED NORMAL ESTIMATION
    # --------------------------------------------------------------------------
    print("\n[STEP 8] Estimating and Orienting Surface Normals...")
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.004, max_nn=30))
    source.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    
    print("-> ACTION: Close window to extract geometric features.")
    o3d.visualization.draw_geometries([source], window_name="Step 8: Oriented Surface Normals", point_show_normal=True)

    #save_processed_scan(source)
    # --------------------------------------------------------------------------
    # STEP 9: FPFH DESCRIPTOR EXTRACTION
    # --------------------------------------------------------------------------
    print("\n[STEP 9] Computing FPFH Descriptors...")
    source_fpfh = extract_fpfh_features(source, VOXEL_SIZE, is_source=True)
    target_fpfh = extract_fpfh_features(pristine_target, VOXEL_SIZE, is_source=False)

    # --------------------------------------------------------------------------
    # STEP 10: RANSAC GLOBAL COARSE ALIGNMENT
    # --------------------------------------------------------------------------
    print("\n[STEP 10] Running RANSAC Global Alignment...")
     # This calculates the center of mass of both files and jumps the CAD model
    # straight onto the scan so RANSAC doesn't have to search across empty space.
    # FIX: Explicitly calculate and capture the initial coarse translation vector
    translation_vec = source.get_center() - pristine_target.get_center()
    
    # Construct the 4x4 homogenous transformation matrix for this initial jump
    T_init = np.eye(4)
    T_init[:3, 3] = translation_vec  # Inject translation vector into the rightmost column
    
    # Physically translate the CAD model into the scan's neighborhood
    pristine_target.translate(translation_vec)
    distance_threshold = 0.005  # 5mm
    normal_cos_threshold = np.cos(np.radians(15.0))
    start_time = time.time()
    
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, pristine_target, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_cos_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000)
    )
    
    print(f"-> RANSAC completed in {time.time() - start_time:.2f}s (Fitness: {ransac_result.fitness:.4f}) | Inlier RMSE: {ransac_result.inlier_rmse:.4f}")
    draw_registration_step(source, pristine_target, ransac_result.transformation, "Step 10: RANSAC Coarse Alignment")

    # --------------------------------------------------------------------------
    # STEP 11: MULTI-SCALE POINT-TO-PLANE ICP REFINEMENT
    # --------------------------------------------------------------------------
    print("\n[STEP 11] Running Multi-Scale Local ICP...")
    icp_thresholds = [0.012, 0.008, 0.004] # 12mm, 8mm, 4mm
    T_current = ransac_result.transformation
    start_time = time.time()

    for i, threshold in enumerate(icp_thresholds):
        print(f"   -> Scale Stage {i+1} (Search Radius: {threshold*1000:.1f}mm)...")
        icp_result = o3d.pipelines.registration.registration_icp(
            source, pristine_target, threshold, 
            init=T_current,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        T_current = icp_result.transformation

    print(f"-> ICP completed in {time.time() - start_time:.2f}s (Fitness: {icp_result.fitness:.4f}) | Inlier RMSE: {icp_result.inlier_rmse:.4f}")
    print("\n[+] Final Optimized 6D Pose Transformation Matrix:\n", T_current)
    draw_registration_step(source, pristine_target, T_current, "Step 11: Multi-Scale ICP Refinement")


    print("[+] Baseline Pose Decoded. Starting Real-Time Visual Tracking Loop...")

   # --------------------------------------------------------------------------
    # STEP 12: ANCHORED REAL-TIME 6DOF POSE TRACKING LOOP
    # --------------------------------------------------------------------------
    print("\n[+] Baseline Pose Decoded. Starting Anchored Real-Time Tracking Loop...")
    print("[*] Tracking Loop Engaged. Move the gearbox slowly. Close the window or press ESC to exit.")
    
    # Waken the camera hardware back up cleanly for the tracking phase
    pipeline.start(config)
    
    # Initialize the Open3D Active Rendering Window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Anchored Real-Time 6DOF Gearbox Tracker", width=1024, height=768)
    
    # Create an empty placeholder point cloud for the incoming live streaming frames
    live_source = o3d.geometry.PointCloud()
    vis.add_geometry(live_source)

    
    
    # # CRITICAL FIX: Create a pristine tracking anchor that NEVER gets mutated by .transform()
    tracking_anchor = copy.deepcopy(pristine_target.voxel_down_sample(voxel_size=VOXEL_SIZE * 3))
   
    # Initialize our absolute tracking matrix relative to the anchor position
    T_anchor_to_camera = np.linalg.inv(T_current)
    
    # Create the visualization container that will be passed to the UI renderer
    tracked_target = copy.deepcopy(tracking_anchor)
    tracked_target.paint_uniform_color([0, 0.651, 0.929])  # Cyan CAD Model
    vis.add_geometry(tracked_target)
    framesCount = 0
    timeStart = time.time()
    timeTester = 0
    
    
    try:
        while True:
            # 1. Pull continuous real-time frames from the active camera pipeline
            timeElapsed = time.time() - timeStart
            if timeElapsed > 5.0:
                fps = framesCount / timeElapsed
                print(f"-> Real-Time Tracking FPS: {fps:.2f}")
                timeStart = time.time()
                framesCount = 0
            framesCount += 1
            timeTester = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            
            
            if not depth_frame:
                continue
                
            depth_image = np.asanyarray(depth_frame.get_data())
            print(f"-> Frame Capture Time: {time.time() - timeTester:.4f}s")
            
            # 2. Transform the active depth matrix into an Open3D point cloud structure
            timeTester = time.time()
            depth_img_o3d = o3d.geometry.Image(depth_image)
            new_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth=depth_img_o3d, intrinsic=INTRINSICS, depth_scale=1000.0, depth_trunc=0.8 # changing depth_trunc from 2.0 to 0.8 for closer range tracking
            )
            print(f"-> Point Cloud Projection Time: {time.time() - timeTester:.4f}s")
            
            # 3. High-speed spatial preprocessing to preserve frame rate
            timeTester = time.time()
            new_pcd = new_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE * 3)  # Slightly larger voxel for speed
            print(f"-> Voxel Downsampling Time: {time.time() - timeTester:.4f}s")

            timeTester = time.time()
            try:
                # Fast RANSAC table strip to isolate the moving target object
                _, inliers = new_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=20) # dropped from 50 to 20 for performance
                new_pcd = new_pcd.select_by_index(inliers, invert=True)
            except:
                pass

            print(f"-> Table Plane Segmentation Time: {time.time() - timeTester:.4f}s")
            timeTester = time.time()
                
            # new_pcd, _ = new_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5) #removed this for performance
            new_pcd.paint_uniform_color([1, 0.706, 0])  # Yellow Scan Data
            
            # Swap data vectors out inside the visualizer thread
            live_source.points = new_pcd.points
            vis.update_geometry(live_source)
            print(f"-> Update UI Geometry Time: {time.time() - timeTester:.4f}s")
            
            # 4. Run Anchored Model-to-Frame ICP tracking against the live stream
            if len(live_source.points) > 100:
                # Generate high-speed temporary normals for the point-to-plane calculations
                timeTester = time.time()
                live_source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=12)) # increased radius from 0.006 to 0.012 and decreased max_nn from 20 to 12 for speed
                print(f"-> Normal Estimation Time: {time.time() - timeTester:.4f}s")
                timeTester = time.time()

                # We track the unmutated ANCHOR directly to the new frame, 
                # seeding it with the previous frame's successful pose matrix.
                track_result = o3d.pipelines.registration.registration_icp(
                    tracking_anchor, live_source, 0.008,  # Slightly opened to 8mm for dynamic motion
                    init=T_anchor_to_camera,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
                )
                print(f"-> ICP Tracking Time: {time.time() - timeTester:.4f}s (Fitness: {track_result.fitness:.4f})")
                timeTester = time.time()
                
                # Overwrite the absolute tracking matrix directly (No delta compounding multiplication needed!)
                T_anchor_to_camera = track_result.transformation
                
                # Safe Reset: Overwrite visualizer coordinates from the anchor to prevent memory leaking drift
                tracked_target.points = o3d.utility.Vector3dVector(np.array(tracking_anchor.points))
                if tracking_anchor.has_normals():
                    tracked_target.normals = o3d.utility.Vector3dVector(np.array(tracking_anchor.normals))
                
                # Snap the visualizer instantly to the absolute position calculated
                tracked_target.transform(T_anchor_to_camera)
                vis.update_geometry(tracked_target)
                
                # Calculate absolute CAD space (origin) to the live camera frame matrix
                T_cad_to_camera = T_anchor_to_camera @ T_init
                # print("\n-> Absolute 6DOF Pose Matrix (CAD to Camera):\n", T_cad_to_camera)
                
            # 5. Flush frame events to UI layer and poll for manual exit sequences
            if not vis.poll_events():
                break
            vis.update_renderer()

            print(f"-> Updating UI and Updating Matrix Time: {time.time() - timeTester:.4f}s")
            
            # Prevent operational CPU thread lock-ups
            time.sleep(0.01)
            
    finally:
        print("\n[*] Exiting tracking loop. Cleaning up...")
        pipeline.stop()
        vis.destroy_window()
        print("[*] Tracking pipeline terminated cleanly.")
if __name__ == "__main__":
    main()