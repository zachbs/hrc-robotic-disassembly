import open3d as o3d
import numpy as np
import copy
import time

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS (ALIGNED WITH FULL PIPELINE)
# ==============================================================================
VOXEL_SIZE = 0.0025  # 2.5mm Voxel Grid
DEPTH_TRUNC = 0.8    # 80cm depth truncation limit

# Ground truth transformation matrix for verification
T_GROUND_TRUTH = np.array([
    [ 0.92122675,  0.2779421 , -0.2721938 , -0.12516987],
    [ 0.37365798, -0.82689225,  0.42027244, -0.02370369],
    [-0.10826353, -0.4888736 , -0.86561054,  0.77498699],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

def draw_registration_step(source, target, transformation, window_name):
    """Helper to visualize alignment steps with consistent coloring."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Yellow = Moving RealSense Scan, Cyan = Static CAD Target
    source_temp.paint_uniform_color([1, 0.706, 0])      
    target_temp.paint_uniform_color([0, 0.651, 0.929])  
    
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def extract_fpfh_features(pcd, voxel_size, is_source=False):
    """Computes geometric surface normals consistently and extracts FPFH descriptors."""
    if is_source:
        # Source (Camera Scan): Always compute fresh normals matching camera perspective
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    else:
        # Target (CAD Mesh): Compute dynamically if missing
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def compute_pose_error(T_est, T_gt):
    """Calculates geodesic rotation error (degrees) and translation error (meters)."""
    R_est = T_est[0:3, 0:3]
    R_gt = T_gt[0:3, 0:3]
    R_relative = np.dot(R_est.T, R_gt)
    
    trace = np.trace(R_relative)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    t_est = T_est[0:3, 3]
    t_gt = T_gt[0:3, 3]
    trans_error_m = np.linalg.norm(t_est - t_gt)
    
    return angle_deg, trans_error_m

def run_visual_pipeline():
    # --- FILE PATH CONFIGURATIONS ---
    PCD_FILE_PATH = "captured_scans/centered_cad_target.pcd"  # CAD model to align with the scan
    PCD_SCAN_PATH = "captured_scans/gearbox_scan_20260616_192600.pcd"  # Preprocessed scan point cloud

    # --- STEP 1: LOAD RAW REALSENSE SCAN & CAD TARGET ---
    print("\n[STEP 1] Loading Raw RealSense Scan & CAD Target...")
    pcd = o3d.io.read_point_cloud(PCD_SCAN_PATH)
    target = o3d.io.read_point_cloud(PCD_FILE_PATH)
    
    
    
 

    # --- STEP 2: VOXEL DOWNSAMPLING ---
    print("\n[STEP 2] Downsampling Point Clouds...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    target = target.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    print(f"-> Downsampled scan to {len(pcd.points)} points.")
    print(f"-> Downsampled CAD target to {len(target.points)} points.")

    # --- STEP 3: TABLE PLANE SEGMENTATION ---
    # print("\n[STEP 3] Removing Dominant Table Plane...")
    # try:
    #     plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=200)
    #     pcd = pcd.select_by_index(inliers, invert=True)
    #     print(f"-> After table removal, scan has {len(pcd.points)} points.")
    # except Exception as e:
    #     print(f"-> Plane segmentation skipped or failed: {e}")

    # --- STEP 4: OUTLIER REMOVAL ---
    print("\n[STEP 4] Removing Statistical Outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
    print(f"-> After statistical outlier removal, scan has {len(pcd.points)} points.")

    source = pcd
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])

    # --- STEP 5: CAMERA-ORIENTED NORMAL ESTIMATION ---
    print("\n[STEP 5] Estimating Surface Normals...")
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))
    source.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    
    draw_registration_step(source, target, np.identity(4), "Step 5: Initial Unaligned State")

    # --- STEP 6: FPFH FEATURE EXTRACTION & GLOBAL RANSAC ---
    print("\n[STEP 6] Computing FPFH Features & Running Global RANSAC...")
    source_fpfh = extract_fpfh_features(source, VOXEL_SIZE, is_source=True)
    target_fpfh = extract_fpfh_features(target, VOXEL_SIZE, is_source=False)


    # target.translate(source.get_center() - target.get_center())
    distance_threshold = VOXEL_SIZE * 1.5  # 0.00375m
    normal_cos_threshold = np.cos(np.radians(45.0))
    
    start_time = time.time()
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.88),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_cos_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000)
    )
    print(f"-> RANSAC completed in {time.time() - start_time:.2f}s (Fitness: {ransac_result.fitness:.4f}) | RMSE: {ransac_result.inlier_rmse:.4f}")
    draw_registration_step(source, target, ransac_result.transformation, "Step 6: Coarse Global RANSAC Output")

    # --- STEP 7: MULTI-SCALE POINT-TO-PLANE ICP REFINEMENT ---
    print("\n[STEP 7] Refining Alignment using Multi-Scale Local ICP...")
    icp_thresholds = [0.012, 0.008, 0.004]  # 12mm, 8mm, 4mm
    T_current = ransac_result.transformation
    start_time = time.time()

    for i, threshold in enumerate(icp_thresholds):
        print(f"   -> Scale Stage {i+1} (Search Radius: {threshold*1000:.1f}mm)...")
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, 
            init=T_current,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        T_current = icp_result.transformation

    rot_error, trans_error = compute_pose_error(T_current, T_GROUND_TRUTH)
    print(f"-> ICP completed in {time.time() - start_time:.2f}s (Fitness: {icp_result.fitness:.4f}) | RMSE: {icp_result.inlier_rmse:.4f}")
    print("\nFinal Optimized 4x4 Transformation Matrix:\n", T_current)
    print(f"Rotation Error from Ground Truth: {rot_error:.4f} degrees")
    print(f"Translation Error from Ground Truth: {trans_error:.6f} meters")
    draw_registration_step(source, target, T_current, "Step 7: Final Multi-Scale ICP Output")

if __name__ == "__main__":
    run_visual_pipeline()