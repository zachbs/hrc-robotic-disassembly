import open3d as o3d
import numpy as np
import copy
import time
import os
import itertools

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
PCD_SCAN_PATH = "captured_scans/gearbox_scan_20260616_192600.pcd" 
STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved.stl"

VOXEL_SIZE = 0.001
CAD_SCALE_FACTOR = 0.00168095 
NUM_ITERATIONS_PER_CONFIG = 5  

T_CORRECT = np.array([
    [ 0.29728389,  0.57904716,  0.75916182, -0.14442216],
    [ 0.92787226, -0.36267385, -0.08672223,  0.28355615],
    [ 0.22511188,  0.73018632, -0.64509889,  0.84109389],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# ==============================================================================
# DEFINE REFOCUSED TARGET SEARCH GRID (54 Clean Combinations)
# ==============================================================================
grid_parameters = {
    'mutual_filter': [False],                  # Pre-pruned based on your previous lockout log
    'distance_threshold': [0.005, 0.008, 0.012],
    'ransac_n': [3, 4],                        # 4 forces non-coplanar spatial scaling
    'feature_radius_mult': [5, 8, 12],         # Captures broader macro-geometry contexts
    'normal_angle_deg': [15.0, 30.0, 45.0]
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def extract_fpfh_features(pcd, voxel_size, radius_multiplier, is_target=False):
    if not pcd.has_normals():
        if is_target:
            radius_normal = 0.002
        else:
            radius_normal = 0.004  
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    if not is_target:
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    
    radius_feature = voxel_size * radius_multiplier
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def compute_pose_error(T_est, T_gt):
    R_est = T_est[0:3, 0:3]
    R_gt = T_gt[0:3, 0:3]
    R_relative = np.dot(R_est.T, R_gt)
    trace = np.trace(R_relative)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    t_est = T_est[0:3, 3]
    t_gt = T_gt[0:3, 3]
    trans_error_mm = np.linalg.norm(t_est - t_gt) * 1000.0
    return angle_deg, trans_error_mm

# ==============================================================================
# RUN SWEEP SEQUENCE
# ==============================================================================
def run_robust_grid_search():
    print("=====================================================================")
    print("        STOCHASTIC RANSAC GRID SEARCH (FIXED REFERENCE SNAPSHOT)     ")
    print("=====================================================================")

    if not os.path.exists(PCD_SCAN_PATH):
        print(f"[-] ERROR: Missing scan file at: {PCD_SCAN_PATH}")
        return

    print("\n[*] Initializing point cloud assets...")
    base_source = o3d.io.read_point_cloud(PCD_SCAN_PATH)
    
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    base_target = mesh.sample_points_uniformly(number_of_points=20000)
    base_target.scale(CAD_SCALE_FACTOR, center=base_target.get_center())
    base_target = base_target.voxel_down_sample(voxel_size=VOXEL_SIZE)
    base_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.004, max_nn=30))
    #base_target.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    
    # Clean center-to-center initialization without the offset error
    base_target.translate(base_source.get_center() - base_target.get_center())

    keys, values = zip(*grid_parameters.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_configs = len(experiments)
    
    print(f"[+] Total Configurations: {total_configs} | Total trials to execute: {total_configs * NUM_ITERATIONS_PER_CONFIG}")
    
    results_log = []
    best_robust_combo = None
    min_avg_rotation_error = float('inf')

    start_search_time = time.time()

    for idx, config in enumerate(experiments, 1):
        run_rot_errors = []
        run_trans_errors = []
        run_fitness_scores = []
        sample_matrices = []

        dist_thresh = config['distance_threshold']
        cos_normal_thresh = np.cos(np.radians(config['normal_angle_deg']))
        rad_mult = config['feature_radius_mult']
        
        # Pre-compute FPFH for this specific radius setting to maximize speed
        source_fpfh = extract_fpfh_features(base_source, VOXEL_SIZE, rad_mult, is_target=False)
        target_fpfh = extract_fpfh_features(base_target, VOXEL_SIZE, rad_mult, is_target=True)
        
        for trial in range(NUM_ITERATIONS_PER_CONFIG):
            source_eval = copy.deepcopy(base_source)
            target_eval = copy.deepcopy(base_target)
            
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_eval, target_eval, source_fpfh, target_fpfh, 
                mutual_filter=config['mutual_filter'],
                max_correspondence_distance=dist_thresh,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=config['ransac_n'],
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.90),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(cos_normal_thresh)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 1000)
            )
            
            # Match the multi-stage local refinement from your production script
            icp_thresholds = [0.010, 0.006, 0.002]
            T_current = ransac_result.transformation
            
            for threshold in icp_thresholds:
                icp_result = o3d.pipelines.registration.registration_icp(
                    source_eval, target_eval, threshold, 
                    init=T_current,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                T_current = icp_result.transformation

            rot_err, trans_err = compute_pose_error(T_current, T_CORRECT)
            
            run_rot_errors.append(rot_err)
            run_trans_errors.append(trans_err)
            run_fitness_scores.append(ransac_result.fitness)
            sample_matrices.append(T_current)

        avg_rot_error = np.mean(run_rot_errors)
        avg_trans_error = np.mean(run_trans_errors)
        avg_ransac_fitness = np.mean(run_fitness_scores)
        std_rot_error = np.std(run_rot_errors)

        # Log to structural metrics tracking container
        log_entry = copy.deepcopy(config)
        log_entry['avg_rot_error'] = avg_rot_error
        log_entry['avg_trans_error'] = avg_trans_error
        log_entry['avg_fitness'] = avg_ransac_fitness
        log_entry['std_rot_error'] = std_rot_error
        results_log.append(log_entry)
        
        if idx % 3 == 0 or idx == total_configs:
            print(f"   -> Configs: {idx}/{total_configs} | Current Baseline Best Avg Rot Error: {min_avg_rotation_error:.3f}°")

        # Save via true deepcopy to eliminate data leakage across iterations
        if avg_rot_error < min_avg_rotation_error:
            min_avg_rotation_error = avg_rot_error
            best_robust_combo = copy.deepcopy(log_entry)
            median_idx = np.argsort(run_rot_errors)[len(run_rot_errors)//2]
            best_robust_combo['matrix'] = copy.deepcopy(sample_matrices[median_idx])

    print(f"\n[+] Execution finalized in {time.time() - start_search_time:.2f} seconds.")
    
    # ==============================================================================
    # STABLE EXPORT MATRIX COMPILATION
    # ==============================================================================
    print("\n=====================================================================")
    print("               VERIFIED OPTIMAL REPEATABILITY PROFILE                ")
    print("=====================================================================")
    print(f"[+] WINNING PARAMETERS PROFILE (STABLE OVER {NUM_ITERATIONS_PER_CONFIG} RUNS):")
    print(f"    -> MEAN Geodesic Rotation Error:   {best_robust_combo['avg_rot_error']:.5f} degrees")
    print(f"    -> Rotation Variance Std Dev:      {best_robust_combo['std_rot_error']:.5f} degrees")
    print(f"    -> MEAN Euclidean Translation Err: {best_robust_combo['avg_trans_error']:.5f} mm")
    print(f"    -> MEAN RANSAC Baseline Fitness:   {best_robust_combo['avg_fitness']:.4f}")
    print("\n[+] REPEATABLE CONFIGURATION VALUES:")
    print(f"    -> mutual_filter:                  {best_robust_combo['mutual_filter']}")
    print(f"    -> distance_threshold:             {best_robust_combo['distance_threshold']}")
    print(f"    -> ransac_n:                       {best_robust_combo['ransac_n']}")
    print(f"    -> feature_radius_mult:            {best_robust_combo['feature_radius_mult']} (Radius: {VOXEL_SIZE * best_robust_combo['feature_radius_mult'] * 1000:.1f}mm)")
    print(f"    -> normal_angle_deg:               {best_robust_combo['normal_angle_deg']} degrees")
    print("=====================================================================")
    
    source_verify = copy.deepcopy(base_source)
    target_verify = copy.deepcopy(base_target)
    source_verify.paint_uniform_color([1, 0.706, 0])      
    target_verify.paint_uniform_color([0, 0.651, 0.929])  
    target_verify.transform(best_robust_combo['matrix'])
    o3d.visualization.draw_geometries([source_verify, target_verify], window_name="True Optimal Match Verification")

if __name__ == "__main__":
    run_robust_grid_search()