import open3d as o3d
import numpy as np
import copy
import time
import os

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Update these paths to match your saved files!
PCD_SCAN_PATH = "captured_scans/gearbox_scan_20260616_192600.pcd" 
STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved-Fusion.stl"
# STL_FILE_PATH = "LowerLidV2.stl"

VOXEL_SIZE = 0.001
CAD_SCALE_FACTOR = 0.00087963  # Your unified assembly scaling factor for non-removed gearbox STL model
CAD_SCALE_FACTOR = 0.00168095
# CAD_SCALE_FACTOR = 0.067

# Ground truth matrix from your lab calibration to verify accuracy
T_GROUND_TRUTH = np.array([
    [ 0.29728389,  0.57904716,  0.75916182, -0.14442216],
    [ 0.92787226, -0.36267385, -0.08672223,  0.28355615],
    [ 0.22511188,  0.73018632, -0.64509889,  0.84109389],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def draw_registration_step(source, target, transformation, window_name):
    """Helper to visualize alignment steps with consistent coloring."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Yellow = Moving RealSense Scan, Cyan = Static Ideal CAD Model (.stl)
    source_temp.paint_uniform_color([1, 0.706, 0])      
    target_temp.paint_uniform_color([0, 0.651, 0.929])  
    
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

# ==============================================================================
# OFFLINE EVALUATION PIPELINE
# ==============================================================================
def run_offline_pipeline():
    print("=====================================================================")
    print("      OFFLINE GEOMETRIC REGISTRATION & EVALUATION PROTOTYPE          ")
    print("=====================================================================")

    # --- STEP 1: LOAD PRE-PROCESSED SCAN AND RAW CAD ---
    if not os.path.exists(PCD_SCAN_PATH):
        print(f"[-] CRITICAL ERROR: Could not find scan file at: {PCD_SCAN_PATH}")
        print("[*] Please check your timestamp filename inside 'captured_scans/' directory.")
        return

    print("\n[STEP 1] Loading Pre-Processed PCD Scan & CAD Assembly...")
    # Load the source point cloud (your real-world scan containing oriented normals)
    source = o3d.io.read_point_cloud(PCD_SCAN_PATH)
    print(f"-> Source Scan Loaded: {len(source.points)} points (Normals Present: {source.has_normals()})")

    # Load the raw STL mesh and turn it into a comparable point cloud target
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    mesh.compute_vertex_normals()  # Generates uniform outward-facing vectors
    target = mesh.sample_points_uniformly(number_of_points=30000)
    print(f"-> Target CAD Mesh sampled into {len(target.points)} raw target points.")

    # --- STEP 2: CAD PRE-PROCESSING (SCALING & DOWNSAMPLING) ---
    print("\n[STEP 2] Pre-processing CAD Model to match Reality...")
    # Apply your specific assembly scaling factor
    target.scale(CAD_SCALE_FACTOR, center=target.get_center())
    target.paint_uniform_color([0.6, 0.6, 0.6])
    
    print("[*] Clearing geometry history and forcing fresh CAD normals...")
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=30))
    
    # 2. FIX: Walk a local neighborhood graph to force all normal vectors 
    # to point in a single, fluidly continuous direction across all faces
    target.orient_normals_consistent_tangent_plane(k=15)

    target = target.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print("-> Press 'N' on your keyboard once the window opens to toggle normal lines visible!")
    print("-> Use '[' and ']' to change the length of the normal lines if they are too small/large.")
    o3d.visualization.draw_geometries([target], point_show_normal=False)
        # ==============================================
    # print(f"-> Target CAD processed: {len(target.points)} points remaining.")

    # --- STEP 3: INITIAL TRANS-COARSE SNAP ---
    print("\n[STEP 3] Jumping CAD model center to Scan center of mass...")
    # Jump the CAD model straight onto the scan so RANSAC starts at the 99-yard line
    target.translate(source.get_center() - target.get_center())

    # Visual Diagnostic Check before registration math executes
    cad_extent = target.get_max_bound() - target.get_min_bound()
    scan_extent = source.get_max_bound() - source.get_min_bound()
    print("\n[SCALE DIAGNOSTIC]")
    print(f"-> CAD Dimensions (Meters): {cad_extent}")
    print(f"-> Scan Dimensions (Meters): {scan_extent}")
    print("-> ACTION: Close visualizer window to initiate Feature Extraction.")
    source.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([source], point_show_normal=True)

    draw_registration_step(source, target, np.identity(4), "Step 3: Coarse Centralized Starting State")


    # --- STEP 4: FEATURE EXTRACTION ---
    print("\n[STEP 4] Computing FPFH Feature Histograms...")
    start_feat = time.time()
    source_fpfh = extract_fpfh_features(source, VOXEL_SIZE, is_source=True)
    target_fpfh = extract_fpfh_features(target, VOXEL_SIZE)
    print(f"-> FPFH extraction complete in {time.time() - start_feat:.3f}s.")

    fitnessThreshold = 0.95
    is_good_enough = False
    icp_result = None
    T_current = np.identity(4)

    while not is_good_enough:



        # --- STEP 5: GLOBAL RANSAC COARSE REGISTRATION ---
        print("\n[STEP 5] Running RANSAC Global Alignment...")
        distance_threshold = 0.005
        start_ransac = time.time()
        normal_cos_threshold = np.cos(np.radians(15.0))
        
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, mutual_filter=True,
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
        print(f"-> RANSAC Finished in {time.time() - start_ransac:.2f}s")
        print(f"-> Coarse Fitness Score: {ransac_result.fitness:.4f}")
        print(f"-> Coarse Inlier RMSE:   {ransac_result.inlier_rmse:.6f}")
        print("-> ACTION: Close visualizer window to begin step-by-step Local ICP loops.")
        draw_registration_step(source, target, ransac_result.transformation, "Step 5: Global RANSAC Coarse Output")

        # --- STEP 6: INTERATIVE MULTI-SCALE LOCAL ICP REFINEMENT ---
        print("\n[STEP 6] Initializing Multi-Scale Point-to-Plane ICP Progression...")
        icp_thresholds = [0.012, 0.008, 0.004]  # Correspondence distance thresholds for each ICP stage (meters)
        T_current = ransac_result.transformation
        
        # Loop over every scale threshold stage, showing you the geometric progress live
        for stage_idx, threshold in enumerate(icp_thresholds):
            print(f"\n   Executing ICP Stage {stage_idx + 1}/{len(icp_thresholds)}")
            print(f"   -> Convergence Search Radius Limit: {threshold * 1000:.2f} mm")
            
            start_icp_stage = time.time()
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, threshold, 
                init=T_current,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            # Update transformation matrix for the next scale boundary
            T_current = icp_result.transformation
            
            print(f"   -> Stage Duration: {time.time() - start_icp_stage:.3f}s")
            print(f"   -> Current Overlap Fitness: {icp_result.fitness:.4f}")
            print(f"   -> Current Inlier RMSE:    {icp_result.inlier_rmse:.6f}")
            
            # Display the result of this individual optimization loop stage
            window_title = f"Step 6 - ICP Scale Stage {stage_idx + 1} ({threshold*1000:.1f}mm limit)"
            draw_registration_step(source, target, T_current, window_title)
            if icp_result.fitness >= fitnessThreshold:
                print(f"   -> Fitness threshold of {fitnessThreshold} reached. Ending ICP progression early.")
                is_good_enough = True
            else:
                print(f"   -> Fitness below threshold. Proceeding to next ICP scale stage.")
                break

    # --- STEP 7: ACCURACY EVALUATION AGAINST GROUND TRUTH ---
    print("\n[STEP 7] Final Pose Error Metric Compilation...")
    rot_error, trans_error = compute_pose_error(T_current, T_GROUND_TRUTH)
    
    print("\n=====================================================================")
    print("                      FINAL ALIGNMENT REPORT                         ")
    print("=====================================================================")
    print(f"[+] Final Optimization Overlap Fitness: {icp_result.fitness:.4f}")
    print(f"[+] Final Optimization Inlier RMSE:    {icp_result.inlier_rmse:.6f}")
    print(f"[+] GEODESIC ROTATION ERROR:            {rot_error:.4f} degrees")
    print(f"[+] EUCLIDEAN TRANSLATION ERROR:         {trans_error * 1000:.4f} mm ({trans_error:.6f} meters)")
    print("=====================================================================")
    
    print("\n[+] Final Optimized 4x4 Transformation Matrix (T):\n", T_current)
    draw_registration_step(source, target, T_current, "Step 7: Final Validated 6D Coordinate Pose Alignment")

if __name__ == "__main__":
    run_offline_pipeline()