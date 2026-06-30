import open3d as o3d
import numpy as np
import copy
import time

def draw_registration_step(source, target, transformation, window_name):
    """Helper to visualize alignment steps with consistent coloring."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Yellow = Moving RealSense Scan, Cyan = Static Ideal CAD Model (.stl)
    source_temp.paint_uniform_color([1, 0.706, 0])      
    target_temp.paint_uniform_color([0, 0.651, 0.929])  
    
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def extract_fpfh_features(pcd, voxel_size):
    """Computes geometric surface normals and FPFH feature descriptors."""
    radius_normal = 0.004  
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def compute_pose_error(T_est, T_gt):
    """Calculates geodesic rotation error (degrees) and translation error (meters)."""
    # 1. Rotation Error via matrix trace cross-multiplication
    R_est = T_est[0:3, 0:3]
    R_gt = T_gt[0:3, 0:3]
    R_relative = np.dot(R_est.T, R_gt)
    
    trace = np.trace(R_relative)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # 2. Translation Error (Euclidean distance)
    t_est = T_est[0:3, 3]
    t_gt = T_gt[0:3, 3]
    trans_error_m = np.linalg.norm(t_est - t_gt)
    
    return angle_deg, trans_error_m

# Your verified correct transformation matrix (from your raw, un-transformed scan)
T_GROUND_TRUTH = np.array([[ 0.92122675,  0.2779421 , -0.2721938 , -0.12516987],
 [ 0.37365798, -0.82689225,  0.42027244, -0.02370369],
 [-0.10826353, -0.4888736 , -0.86561054,  0.77498699],
 [ 0.        ,  0.        ,  0.        ,  1.        ]
])

def run_visual_pipeline():
    # --- FILE PATH CONFIGURATIONS ---
    # Update these string paths to point to your local assets
    STL_FILE_PATH = "nonScaledFullGearbox.stl"
    NUMPY_SCAN_PATH = "realsense_lab_data/frame_0001_cropped_depth.npy"  # Path to your saved RealSense depth map in .npy format

   # --- STEP 1: LOAD RAW REALSENSE SCAN & CAD TARGET ---
    print("\n[STEP 1] Loading Raw RealSense Scan & CAD Target...")
    
    # 1. Load the RealSense Numpy Depth Map
    realsense_depth = np.load(NUMPY_SCAN_PATH)
    print(f"-> Target data array shape from disk: {realsense_depth.shape}, Type: {realsense_depth.dtype}")
    
    # Convert the NumPy array into an Open3D Image object
    depth_image = o3d.geometry.Image(realsense_depth)
    
    # Define your camera intrinsics. 
    # These are typical default values for an Intel RealSense D435 at 640x480 resolution.
    # For maximum physical accuracy later, replace these with your exact camera calibration!
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=640, 
        height=480, 
        fx=609.3075561523438,  # Corrected from 425.0
        fy=608.9049072265625,  # Corrected from 425.0  # Focal length Y
        cx=326.06304931640625,  # Principal point X
        cy=249.21212768554688   # Principal point Y
    )
    
    # Project the 2D depth pixels into a 3D Point Cloud
    raw_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_image,
        intrinsic=intrinsics,
        depth_scale=1000.0,  # Scales uint16 millimeters to Open3D meters
        depth_trunc=3.0        # Cuts off any background noise past 3 meters
    )
    
    # 2. Load the STL mesh and convert it to a point cloud
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    raw_target_cad = mesh.sample_points_uniformly(number_of_points=20000)

    raw_target_cad.scale(0.00087963, center=raw_target_cad.get_center())
    # This calculates the center of mass of both files and jumps the CAD model
    # straight onto the scan so RANSAC doesn't have to search across empty space.
    raw_target_cad.translate(raw_pcd.get_center() - raw_target_cad.get_center())

    # Diagnostic check for bounding box sizes
    cad_extent = raw_target_cad.get_max_bound() - raw_target_cad.get_min_bound()
    scan_extent = raw_pcd.get_max_bound() - raw_pcd.get_min_bound()
    
    print(f"\n[SCALE DIAGNOSTIC]")
    print(f"-> CAD Model Dimensions (X, Y, Z): {cad_extent}")
    print(f"-> RealSense Scan Dimensions (X, Y, Z): {scan_extent}")
    
    print(f"-> Unpacked RealSense depth map into 3D point cloud with {len(raw_pcd.points)} points.")
    print(f"-> Sampled CAD Model (.stl) contains {len(raw_target_cad.points)} points.")
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([raw_pcd], window_name="Step 1: Generated 3D Point Cloud from Depth Image")
    
    # ⚠️ THE METRIC UNIT CRITICAL TRAP:
    # CAD programs usually export STLs in millimeters, while RealSense data is in meters.
    # If your CAD model looks 1000x larger than your workspace scan, uncomment the line below:
    # raw_target_cad.scale(0.001, center=raw_target_cad.get_center())

    # --- STEP 2: VOXEL DOWNSAMPLING ---
    print("\n[STEP 2] Downsampling Point Clouds...")
    voxel_size = 0.001  # 5cm grid cells (Adjust lower to 0.02 or 0.01 for small CAD parts)
    
    downsampled_pcd = raw_pcd.voxel_down_sample(voxel_size=voxel_size)
    target_downsampled = raw_target_cad.voxel_down_sample(voxel_size=voxel_size)
    
    downsampled_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([downsampled_pcd], window_name="Step 2: Downsampled RealSense Scene")

    # --- STEP 3: SURFACE NORMAL ESTIMATION ---
    print("\n[STEP 3] Estimating Surface Normals...")
    radius_normal = 0.004  # 4mm radius for normal estimation (Adjust based on point cloud density)
    
    # Compute normals across both point cloud models while intact to protect edge context
    downsampled_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_downsampled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    print("-> Surface orientation vectors calculated.")
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([downsampled_pcd], 
                                      window_name="Step 3: RealSense Scene Surface Normals", 
                                      point_show_normal=True)

    # --- STEP 4: CLEAN CAD MODEL SELECTION ---
    print("\n[STEP 4] Locking Down Clean CAD Model (Target)...")
    target = target_downsampled
    target.paint_uniform_color([0, 0.651, 0.929]) 
    
    print(f"-> CAD Target prepared with {len(target.points)} points.")
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([target], window_name="Step 4: Downsampled CAD Target Tracking Geometry")

    # --- STEP 5: ISOLATING THE REAL OBJECT FROM SCAN ---
    print("\n[STEP 5] Cropping Scan ROI & Injecting Initial Displacement...")
    
    # Crop the real environment scan down to where the part physically sits on your table
    # CHANGE THESE BOUNDS to match your real camera coordinate setup!
    # min_bound = np.array([-1, -1, -1])
    # max_bound = np.array([1, 1, 0.9])
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # source = downsampled_pcd.crop(bbox)

    #delete this 
    source = downsampled_pcd
    print(f"-> Pre-filtered scene crop contains {len(source.points)} points.")
    
    # # Strip the flat table plane away from the sensor scan
    try:
        plane_model, inliers = source.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=200)
        source = source.select_by_index(inliers, invert=True)
        print(f"-> After removing dominant table plane, source has {len(source.points)} points.")
    except Exception as e:
        print(f"-> Plane segmentation skipped or failed: {e}")

    # Apply Statistical Outlier Removal to strip away remaining airborne point noise
    source, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    print(f"-> After removing statistical outliers, source has {len(source.points)} points.")

    # Inject a known initial simulated displacement offset to test alignment convergence performance
    # (X = 5cm, Y = -3cm, Z = 2cm)
    T_init = np.array([[ 0.866, -0.500,  0.000,  0.05],
                       [ 0.500,  0.866,  0.000, -0.03],
                       [ 0.000,  0.000,  1.000,  0.02],
                       [ 0.000,  0.000,  0.000,  1.000]])
    source.transform(T_init)

    cad_extent = raw_target_cad.get_max_bound() - raw_target_cad.get_min_bound()
    scan_extent = source.get_max_bound() - source.get_min_bound()

    print(f"\n[SCALE DIAGNOSTIC]")
    print(f"-> CAD Model Dimensions (X, Y, Z): {cad_extent}")
    print(f"-> RealSense Scan Dimensions (X, Y, Z): {scan_extent}")
    
    print(f"-> Unpacked RealSense depth map into 3D point cloud with {len(raw_pcd.points)} points.")
    print(f"-> Sampled CAD Model (.stl) contains {len(raw_target_cad.points)} points.")
    
    print("-> ACTION: Close window to launch Global RANSAC.")
    draw_registration_step(source, target, np.identity(4), "Step 5: Initial Unaligned State (Yellow=RealScan, Cyan=CAD Model)")

    # --- STEP 6: GLOBAL RANSAC REGISTRATION ---
    print("\n[STEP 6] Computing Features & Running Global RANSAC...")
    source_fpfh = extract_fpfh_features(source, voxel_size)
    target_fpfh = extract_fpfh_features(target, voxel_size)
    distance_threshold = 0.005
    
    start_time = time.time()

    
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 500)
    )
    finish_time = time.time()
    print(f"-> RANSAC completed in {finish_time - start_time:.2f} seconds with fitness: {ransac_result.fitness:.4f} and inlier RMSE: {ransac_result.inlier_rmse:.4f}")
    print("-> RANSAC Coarse alignment complete.")
    print("-> ACTION: Close window to launch fine-tuning Local ICP.")
    draw_registration_step(source, target, ransac_result.transformation, "Step 6: Coarse Global RANSAC Alignment Output")

    # --- STEP 7: LOCAL ICP REFINEMENT ---
    print("\n[STEP 7] Refining Alignment using Local Point-to-Plane ICP...")
    icp_distance_threshold = 0.005
    
    start_time = time.time()
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, icp_distance_threshold, 
        init=ransac_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    finish_time = time.time()
    rot_error, trans_error = compute_pose_error(icp_result.transformation, T_GROUND_TRUTH)
    print(f"-> ICP completed in {finish_time - start_time:.2f} seconds with fitness: {icp_result.fitness:.4f} and inlier RMSE: {icp_result.inlier_rmse:.4f}")
    print("-> Local ICP refinement complete.")
    print("\nFinal Optimized 4x4 Transformation Matrix:\n", icp_result.transformation)
    print(f"Pipeline Finished. Overlap Fitness Score: {icp_result.fitness:.4f}")
    print(f"Rotation Error from Ground Truth: {rot_error:.4f} degrees")
    print(f"Translation Error from Ground Truth: {trans_error:.6f} meters")
    draw_registration_step(source, target, icp_result.transformation, "Step 7: Final Precision ICP Alignment Output")

if __name__ == "__main__":
    run_visual_pipeline()