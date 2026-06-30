import open3d as o3d
import numpy as np
import copy
import time

def draw_registration_step(source, target, transformation, window_name):
    """Helper to visualize alignment steps with consistent coloring."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Yellow = Moving Model, Cyan = Static Workspace Scan
    source_temp.paint_uniform_color([1, 0.706, 0])      
    target_temp.paint_uniform_color([0, 0.651, 0.929])  
    
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def extract_fpfh_features(pcd, voxel_size):
    """Computes geometric surface normals and FPFH feature descriptors."""
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def run_visual_pipeline():
    # --- STEP 1: RAW BASE SCAN ---
    print("\n[STEP 1] Loading Raw Base Scan...")
    demo_data = o3d.data.DemoICPPointClouds()
    raw_pcd = o3d.io.read_point_cloud(demo_data.paths[0])
    print(f"-> Raw scan contains {len(raw_pcd.points)} points.")
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([raw_pcd], window_name="Step 1: Raw Base Scan (Full Environment)")

    # --- STEP 2: VOXEL DOWNSAMPLING ---
    print("\n[STEP 2] Downsampling Point Cloud...")
    voxel_size = 0.1  # 10cm grid cells
    downsampled_pcd = raw_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"-> Downsampled scan compressed to {len(downsampled_pcd.points)} points.")
    
    # Paint it gray to see the raw structural grid clearly
    downsampled_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([downsampled_pcd], window_name="Step 2: Voxel Downsampled Point Cloud")

    # --- STEP 3: SURFACE NORMAL ESTIMATION ---
    print("\n[STEP 3] Estimating Surface Normals...")
    radius_normal = voxel_size * 2
    downsampled_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    print("-> Surface orientation vectors calculated.")
    print("-> ACTION: Close the window to proceed. (You can zoom in to see the vector lines)")
    o3d.visualization.draw_geometries([downsampled_pcd], 
                                      window_name="Step 3: Surface Normals Visualized", 
                                      point_show_normal=True)

    # --- STEP 4: CROPPING THE REGION OF INTEREST (ROI) ---
    print("\n[STEP 4] Cropping Point Cloud to Isolate Object (Chair)...")
    min_bound = np.array([2.0, 1.0, 1.0])
    max_bound = np.array([2.8, 2.3, 1.6])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Let's paint the cropped target cyan to signify it's our clean tracking target
    target = downsampled_pcd.crop(bbox)
    target.paint_uniform_color([0, 0.651, 0.929]) 
    print(f"-> ROI Cropped. Isolated object contains {len(target.points)} points.")
    print("-> ACTION: Close the window to proceed.")
    o3d.visualization.draw_geometries([target], window_name="Step 4: Cropped ROI (Isolated Target)")

    # --- STEP 5: INITIAL UNALIGNED STATE ---
    print("\n[STEP 5] Generating Unaligned Source Model (Simulating CAD initial state)...")
    min_bound = np.array([1.8, 1.0, 1.0])
    max_bound = np.array([2.8, 2.5, 1.8])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Let's paint the cropped target cyan to signify it's our clean tracking target
    
    source = downsampled_pcd.crop(bbox)
    # source = copy.deepcopy(target)
    # Inject a massive simulated displacement offset (30 deg tilt + translation)
    T_init = np.array([[ 0.866, -0.500,  0.000,  0.50],
                       [ 0.500,  0.866,  0.000, -0.30],
                       [ 0.000,  0.000,  1.000,  0.20],
                       [ 0.000,  0.000,  0.000,  1.000]])
    source.transform(T_init)
    print(f"-> Pre-filtered crop contains {len(source.points)} points.")
    # Run this on your cropped cloud BEFORE calculating FPFH features
    plane_model, inliers = source.segment_plane(distance_threshold=0.02,
                                                ransac_n=3,
                                                num_iterations=200)

    # Select everything that is NOT the flat table plane
    source = source.select_by_index(inliers, invert=True)
    
    # source = copy.deepcopy(target)

    print(f"-> After removing dominant plane, source has {len(source.points)} points.")

    # 2. NEW: Apply Statistical Outlier Removal to clear non-planar clutter
    # nb_neighbors: considers 20 surrounding points to calculate average distance
    # std_ratio: lower means more aggressive filtering of stray points

    source, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

    print(f"-> After removing statistical outliers, source has {len(source.points)} points.")

    
    print("-> ACTION: Close window to launch Global RANSAC.")
    draw_registration_step(source, target, np.identity(4), "Step 5: Initial Unaligned State (Yellow=CAD, Cyan=Scan)")

    # --- STEP 6: GLOBAL RANSAC REGISTRATION ---
    print("\n[STEP 6] q...")
    source_fpfh = extract_fpfh_features(source, voxel_size)
    target_fpfh = extract_fpfh_features(target, voxel_size)
    distance_threshold = voxel_size * 1.2
    
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
    icp_distance_threshold = voxel_size * 0.3
    
    start_time = time.time()

    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, icp_distance_threshold, 
        init=ransac_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    finish_time = time.time()
    print(f"-> ICP completed in {finish_time - start_time:.2f} seconds with fitness: {icp_result.fitness:.4f} and inlier RMSE: {icp_result.inlier_rmse:.4f}")
    print("-> Local ICP refinement complete.")
    print("\nFinal Optimized 4x4 Transformation Matrix:\n", icp_result.transformation)
    print(f"Pipeline Finished. Overlap Fitness Score: {icp_result.fitness:.4f}")
    draw_registration_step(source, target, icp_result.transformation, "Step 7: Final Precision ICP Alignment Output")

if __name__ == "__main__":
    run_visual_pipeline()