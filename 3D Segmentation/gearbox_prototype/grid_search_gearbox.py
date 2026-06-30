import open3d as o3d
import numpy as np
import time
import itertools

# ==========================================
# 1. GROUND TRUTH & SCORING CONFIGURATION
# ==========================================

# Your verified correct transformation matrix (from your raw, un-transformed scan)
T_GROUND_TRUTH = np.array([[ 0.92122675,  0.2779421 , -0.2721938 , -0.12516987],
 [ 0.37365798, -0.82689225,  0.42027244, -0.02370369],
 [-0.10826353, -0.4888736 , -0.86561054,  0.77498699],
 [ 0.        ,  0.        ,  0.        ,  1.        ]
])


WEIGHTS = {
    "fitness": 100.0,       # Maximize alignment point coverage
    "rmse": -5000.0,        # Minimize distance error tightly
    "time": 0.0,            # 0.0: Disabled so it doesn't favor broken/fast configurations
    "rot_deg": -10.0,       # Penalize rotation error away from ground truth
    "trans_m": -2000.0,     # Penalize translation drift (meters)
    "flip_penalty": -50000.0 # Extreme penalty if alignment flips past 15 degrees
}

# The Hyperparameter Grid (Now using absolute physical distances instead of multipliers)
PARAM_GRID = {
    # Fine resolution step sizes to capture thin gearbox side-walls cleanly
    "voxel_size": [0.001, 0.0012, 0.0015, 0.002], 
    "ransac_n": [4],                                
    "edge_length": [0.75, 0.8, 0.9],                        
    # Absolute physical search bounds (in meters) to tolerate the camera's depth noise floor
    "ransac_dist": [0.005, 0.006, 0.008],         
    "icp_dist": [0.003, 0.004, 0.005]                
}

RUNS_PER_COMBO = 3  # Number of iterations to average out RANSAC's stochastic nature

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

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

def extract_fpfh_features(pcd, voxel_size):
    """Computes stable descriptors by widening the normal/feature calculation neighborhoods."""
    # Fixed 4mm neighborhood to smooth out raw sensor jitter/grain when computing surface orientations
    radius_normal = 0.004  
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Fixed 10mm neighborhood to capture distinct geometric transitions across the part
    radius_feature = 0.010 
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

def load_and_preprocess():
    print("-> Initializing Base Data (Loading, Scaling, Cropping)...")
    STL_FILE_PATH = "LowerLidV2.stl"
    NUMPY_SCAN_PATH = "frame_0011_depth.npy"

    raw_pcd = o3d.geometry.PointCloud()
    realsense_depth = np.load(NUMPY_SCAN_PATH)
    depth_image = o3d.geometry.Image(realsense_depth)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 615.0, 615.0, 320.0, 240.0)
    raw_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_image, intrinsic=intrinsics, depth_scale=1000.0, depth_trunc=3.0)
    
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    raw_target_cad = mesh.sample_points_uniformly(number_of_points=20000)
    raw_target_cad.scale(0.0800, center=raw_target_cad.get_center())

    min_bound = np.array([-1, -1, -1])
    max_bound = np.array([1, 1, 0.7])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    source = raw_pcd.crop(bbox)
    
    plane_model, inliers = source.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=200)
    source = source.select_by_index(inliers, invert=True)
    source, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

    # REMOVED: source.transform(T_init) 
    # This was distorting the coordinate system away from your true ground truth matrix.
    
    return source, raw_target_cad

# ==========================================
# 3. THE GRID SEARCH ENGINE
# ==========================================

def run_grid_search():
    base_source, base_target = load_and_preprocess()
    
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n[STARTING PHOENIX GRID SEARCH] Testing {len(combinations)} parameter combinations.")
    print(f"Enforcing fixed physical search bounds to isolate camera noise.\n")
    print(f"{'Combo':<5} | {'Voxel':<6} | {'Edge':<4} | {'RAN_d':<5} | {'ICP_d':<5} | {'Avg Fit':<7} | {'Avg RMSE':<8} | {'Rot Err':<7} | {'Tr Err':<7} | SCORE")
    print("-" * 115)

    best_score = -99999999
    best_params = None
    best_metrics = None

    for idx, params in enumerate(combinations):
        v_size = params["voxel_size"]
        
        source = base_source.voxel_down_sample(voxel_size=v_size)
        target = base_target.voxel_down_sample(voxel_size=v_size)
        
        # Build features using our stabilized feature radii functions
        source_fpfh = extract_fpfh_features(source, v_size)
        target_fpfh = extract_fpfh_features(target, v_size)
        
        # Ensure target cloud has estimated normals populated for the Point-to-Plane ICP step
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.004, max_nn=30))

        ransac_dist = params["ransac_dist"]
        icp_dist = params["icp_dist"]

        run_fitness = []
        run_rmse = []
        run_time = []
        run_rot_err = []
        run_trans_err = []

        for i in range(RUNS_PER_COMBO):
            start_time = time.time()
            
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh, mutual_filter=False,
                max_correspondence_distance=ransac_dist,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=params["ransac_n"],
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(params["edge_length"]),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 500)
            )
            
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, icp_dist, 
                init=ransac_result.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            elapsed = time.time() - start_time
            
            # Calculate actual matrix deviations against Ground Truth
            rot_error, trans_error = compute_pose_error(icp_result.transformation, T_GROUND_TRUTH)
            
            run_fitness.append(icp_result.fitness)
            run_rmse.append(icp_result.inlier_rmse)
            run_time.append(elapsed)
            run_rot_err.append(rot_error)
            run_trans_err.append(trans_error)

        # Average iterations
        avg_fit = np.mean(run_fitness)
        avg_rmse = np.mean(run_rmse)
        avg_time = np.mean(run_time)
        avg_rot = np.mean(run_rot_err)
        avg_trans = np.mean(run_trans_err)

        # Comprehensive spatial penalty score
        score = ((avg_fit * WEIGHTS["fitness"]) 
                 + (avg_rmse * WEIGHTS["rmse"]) 
                 + (avg_time * WEIGHTS["time"])
                 + (avg_rot * WEIGHTS["rot_deg"])
                 + (avg_trans * WEIGHTS["trans_m"]))

        # Hard orientation guardrail
        if avg_rot > 15.0:
            score += WEIGHTS["flip_penalty"]

        print(f"#{idx+1:<4} | {v_size:<6} | {params['edge_length']:<4} | {ransac_dist:<5} | {icp_dist:<5} | {avg_fit:.4f}  | {avg_rmse:.6f} | {avg_rot:.2f}° | {avg_trans*100:.2f}cm | {score:.2f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = (avg_fit, avg_rmse, avg_time, avg_rot, avg_trans)

    # ==========================================
    # 4. RESULTS OUTPUT
    # ==========================================
    print("\n" + "="*50)
    print("🏆 OPTIMAL PARAMETER SETUP FOUND 🏆")
    print("="*50)
    print(f"Score:                {best_score:.2f}")
    print(f"Voxel Size:           {best_params['voxel_size']} m")
    print(f"Edge Length Checker:  {best_params['edge_length']}")
    print(f"Fixed RANSAC Bound:   {best_params['ransac_dist']} m")
    print(f"Fixed ICP Bound:      {best_params['icp_dist']} m")
    print("-" * 50)
    print(f"Average Fitness:      {best_metrics[0]:.4f}")
    print(f"Average RMSE:         {best_metrics[1]:.6f}")
    print(f"Average Execution:    {best_metrics[2]:.4f}s")
    print(f"Final Rotation Error: {best_metrics[3]:.4f}°")
    print(f"Final Trans. Error:   {best_metrics[4]*100:.4f}cm")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_grid_search()