import open3d as o3d
import numpy as np
import copy
import time
import itertools

def extract_fpfh_features(pcd, voxel_size):
    """Computes geometric surface normals and FPFH feature descriptors."""
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def evaluate_pipeline(raw_pcd, voxel_size, ransac_mult, icp_mult, sor_std):
    """Runs the pipeline silently without visual popups and returns performance metrics."""
    start_total_time = time.time()
    
    # 1. Downsample
    downsampled_pcd = raw_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 2. Estimate Normals
    radius_normal = voxel_size * 2
    downsampled_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 3. Crop Target
    bbox_target = o3d.geometry.AxisAlignedBoundingBox(np.array([2.0, 1.0, 1.0]), np.array([2.8, 2.3, 1.6]))
    target = downsampled_pcd.crop(bbox_target)
    
    # 4. Crop Source & Inject Displacement
    bbox_source = o3d.geometry.AxisAlignedBoundingBox(np.array([1.8, 1.0, 1.0]), np.array([2.8, 2.5, 1.8]))
    source = downsampled_pcd.crop(bbox_source)
    
    T_init = np.array([[ 0.866, -0.500,  0.000,  0.50],
                       [ 0.500,  0.866,  0.000, -0.30],
                       [ 0.000,  0.000,  1.000,  0.20],
                       [ 0.000,  0.000,  0.000,  1.000]])
    source.transform(T_init)
    
    # 5. Clean Table Plane
    try:
        plane_model, inliers = source.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=200)
        source = source.select_by_index(inliers, invert=True)
    except:
        return 0.0, 1.0, 99.0 # Return failure scores if geometry is broken
        
    # 6. Statistical Outlier Removal
    source, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=sor_std)
    
    # 7. Global RANSAC
    source_fpfh = extract_fpfh_features(source, voxel_size)
    target_fpfh = extract_fpfh_features(target, voxel_size)
    distance_threshold = voxel_size * ransac_mult
    
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 500) # Capped to keep grid fast
    )
    
    # 8. Local ICP
    icp_distance_threshold = voxel_size * icp_mult
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, icp_distance_threshold, 
        init=ransac_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    total_elapsed_time = time.time() - start_total_time
    return icp_result.fitness, icp_result.inlier_rmse, total_elapsed_time

def run_optimization_search():
    print("[INIT] Loading Base Scan Data...")
    demo_data = o3d.data.DemoICPPointClouds()
    raw_pcd = o3d.io.read_point_cloud(demo_data.paths[0])
    
    # --- HYPERPARAMETER SEARCH SPACE ---
    # 3 * 3 * 3 * 2 = 54 Total Combinations
    voxel_sizes = [0.06, 0.08, 0.10]
    ransac_multipliers = [1.2, 1.5, 1.8]
    icp_multipliers = [0.3, 0.4, 0.5]
    sor_std_ratios = [0.8, 1.2]
    
    # --- NORMALIZATION BOUNDARIES ---
    # Used to scale unbounded values smoothly between 0.0 and 1.0
    MAX_ACCEPTABLE_RMSE = 0.05  # 5cm max deviation allowed before zero points
    MAX_ACCEPTABLE_TIME = 2.0  # 2.0 seconds max before zero points
    
    best_score = -1.0
    best_params = {}
    best_metrics = {}
    
    combinations = list(itertools.product(voxel_sizes, ransac_multipliers, icp_multipliers, sor_std_ratios))
    print(f"[START] Beginning Grid Search optimization across {len(combinations)} combinations...\n")
    
    for i, (v_size, r_mult, i_mult, s_std) in enumerate(combinations):
        fitness, rmse, exec_time = evaluate_pipeline(raw_pcd, v_size, r_mult, i_mult, s_std)
        
        # Calculate individual normalized 0 to 1 components
        # (Higher is better for all component scores)
        fitness_score = fitness
        rmse_score = max(0.0, 1.0 - (rmse / MAX_ACCEPTABLE_RMSE))
        time_score = max(0.0, 1.0 - (exec_time / MAX_ACCEPTABLE_TIME))
        
        # --- THE WEIGHTED SUM SCORING FUNCTION ---
        # 40% RMSE, 30% Speed (Time), 30% Fitness (Overlap)
        total_score = (0.40 * rmse_score) + (0.30 * time_score) + (0.30 * fitness_score)
        
        if total_score > best_score:
            best_score = total_score
            best_params = {
                "voxel_size": v_size,
                "ransac_multiplier": r_mult,
                "icp_multiplier": i_mult,
                "sor_std_ratio": s_std
            }
            best_metrics = {"fitness": fitness, "rmse": rmse, "time": exec_time}
            print(f"-> [NEW LEADER] Combo #{i+1}: Score = {best_score:.4f} (Time: {exec_time:.2f}s, RMSE: {rmse:.4f}, Fitness: {fitness:.4f})")

    # --- PRINT THE CHAMPION CONFIGURATION ---
    print("\n" + "="*50)
    print("        OPTIMIZATION GRID SEARCH COMPLETE")
    print("="*50)
    print(f"🏆 Winning Composite Score: {best_score:.4f}")
    print("\n✨ Best Parameter Configurations:")
    for param, val in best_params.items():
        print(f"   • {param}: {val}")
    print("\n📊 Resulting Performance Metrics:")
    print(f"   • Overlap Fitness Score : {best_metrics['fitness']:.4f}")
    print(f"   • Inlier Precision RMSE : {best_metrics['rmse']:.5f} meters")
    print(f"   • Total Pipeline Speed  : {best_metrics['time']:.4f} seconds")
    print("="*50)

if __name__ == "__main__":
    run_optimization_search()