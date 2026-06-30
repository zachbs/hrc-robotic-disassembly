import open3d as o3d
import numpy as np

# ==========================================
# PASTE YOUR "SLIGHTLY OFF" MATRIX HERE
# ==========================================
T_SLIGHTLY_OFF = np.array([
    [ 0.66290183,  0.69457167, -0.27951987, -0.12999756],
    [ 0.73559834, -0.53465107,  0.41598475, -0.08098942],
    [ 0.13948563, -0.48137141, -0.86534694,  0.77297153],
    [ 0.0,          0.0,          0.0,          1.0        ]
])

def refine_ground_truth():
    print("-> Loading data for high-precision refinement...")
    STL_FILE_PATH = "LowerLidV2.stl"
    NUMPY_SCAN_PATH = "frame_0011_depth.npy"

    # 1. Load Scan
    realsense_depth = np.load(NUMPY_SCAN_PATH)
    depth_image = o3d.geometry.Image(realsense_depth)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 615.0, 615.0, 320.0, 240.0)
    source = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_image, intrinsic=intrinsics, depth_scale=1000.0, depth_trunc=3.0)

    # 2. Load and Scale CAD
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    target = mesh.sample_points_uniformly(number_of_points=50000) # Higher density for perfect refinement
    target.scale(0.0800, center=target.get_center())

    # 3. Clean Scan Data (Crop & Table Removal)
    min_bound = np.array([-1, -1, -1])
    max_bound = np.array([1, 1, 0.7])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    source = source.crop(bbox)
    plane_model, inliers = source.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=200)
    source = source.select_by_index(inliers, invert=True)
    source, _ = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

    # Downsample slightly to smooth out sensor noise
    source = source.voxel_down_sample(voxel_size=0.001)

    # 4. Estimate High-Accuracy Surface Normals
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.004, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.004, max_nn=30))

    print("-> Pulling the matrix into alignment via tight Point-to-Plane ICP...")
    # Use a very narrow search radius (4mm) because we know it's already close
    # This prevents it from snapping to the wrong side walls
    refinement_radius = 0.004 
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, refinement_radius, 
        init=T_SLIGHTLY_OFF,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200,        # Give it plenty of iterations to settle
            relative_fitness=1e-7,    # Extremely tight convergence criteria
            relative_rmse=1e-7
        )
    )

    print("\n" + "="*60)
    print("💎 PERFECTLY REFINED GROUND TRUTH MATRIX FOUND 💎")
    print("="*60)
    print(f"Refinement Fitness: {icp_result.fitness:.4f}")
    print(f"Refinement RMSE:    {icp_result.inlier_rmse:.6f}")
    print("-" * 60)
    print("Copy and paste this exact array into your validation scripts:")
    print(np.array2string(icp_result.transformation, precision=8, suppress_small=True, separator=", "))
    print("="*60 + "\n")

    # Optional Visual Check
    source.transform(icp_result.transformation)
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 0.5,  1])
    print("Showing visual confirmation window... (Close to exit)")
    o3d.visualization.draw_geometries([source, target], window_name="Refinement Verification")

if __name__ == "__main__":
    refine_ground_truth()