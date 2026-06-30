import open3d as o3d
import numpy as np

# ==============================================================================
# 1. PASTE YOUR FOUND MATRIX HERE
# ==============================================================================
# Replace this identity matrix with the exact 4x4 matrix printed at the very end 
# of your autonomous surface normal grid search output.
T_FOUND = np.array( [[ 0.3062235,   0.58072452,  0.75431174, -0.3442777 ],
 [-0.92555785,  0.3669449,   0.09324217, -0.01184142],
 [-0.22264283, -0.7267121,   0.64986129,  0.1521999 ],
 [ 0.,          0.,          0.,          1.        ]])

def verify_alignment():
    print("-> Loading and preprocessing raw data...")
    STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved-Fusion.stl"
    PCD_SCAN_PATH = "captured_scans/gearbox_scan_20260616_192600.pcd" # Optional: If you saved the intermediate point cloud after preprocessing, load it here for a more direct comparison.

    # # 1. Load Raw Scan Depth and Convert to Point Cloud
    # realsense_depth = np.load(NUMPY_SCAN_PATH)
    # depth_image = o3d.geometry.Image(realsense_depth)
    # intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 615.0, 615.0, 320.0, 240.0)
    # source = o3d.geometry.PointCloud.create_from_depth_image(
    #     depth=depth_image, intrinsic=intrinsics, depth_scale=1000.0, depth_trunc=3.0)
    
    source = o3d.io.read_point_cloud(PCD_SCAN_PATH)
    

    # 2. Load CAD Target and Scale Natively
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    target = mesh.sample_points_uniformly(number_of_points=20000)
    target.scale(0.00168095, center=target.get_center())

    # 3. Apply Workspace Crop Box
    # min_bound = np.array([-1, -1, -1])
    # max_bound = np.array([1, 1, 0.7])
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # source = source.crop(bbox)

    # # 4. Segment and Remove Table/Floor Background
    # plane_model, inliers = source.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=200)
    # source = source.select_by_index(inliers, invert=True)

    # # 5. Remove High-Frequency Sensor Outliers
    # source, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

    print("-> Preprocessing Complete.")
    print(f"-> Verified clean scan points remaining: {len(source.points)}")
    
    # 6. Apply your found transformation matrix to the cleaned source cloud
    print("\n-> Applying your found transformation matrix to align the scan...")
    target.translate(source.get_center() - target.get_center())

    source.transform(T_FOUND)

    # 7. Colorize the point clouds for a clear visual comparison
    source.paint_uniform_color([1.0, 0.0, 0.0]) # RED = Your Processed Scan Data
    target.paint_uniform_color([0.0, 0.5, 1.0]) # LIGHT BLUE = CAD Reference Model

    # 8. Launch the interactive Open3D Visualizer Window
    print("\n==============================================")
    print("🖥️  LAUNCHING INTERACTIVE 3D VISUALIZATION Check")
    print("==============================================")
    print("🔴 RED CLOUD:  Your Preprocessed Scan Data")
    print("🔵 BLUE CLOUD: The Target CAD Reference Model")
    print("----------------------------------------------")
    print("🖱️  Controls:")
    print(" - Left-Click + Drag: Rotate the view")
    print(" - Right-Click + Drag: Pan the view")
    print(" - Scroll Wheel: Zoom In / Zoom Out")
    print("----------------------------------------------")
    print("👉 Close the 3D window to end the script.")
    print("==============================================\n")
    
    o3d.visualization.draw_geometries(
        [source, target], 
        window_name="Verification Frame: Does It Snap?",
        width=1280, 
        height=720
    )

if __name__ == "__main__":
    verify_alignment()