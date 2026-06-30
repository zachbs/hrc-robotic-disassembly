import open3d as o3d
import numpy as np

def run_diagnostics():
    print("==================================================")
    print("🔍 RUNNING GEOMETRIC PIPELINE DIAGNOSTICS 🔍")
    print("==================================================\n")

    # 1. Load Data
    STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved.stl"
    NUMPY_SCAN_PATH = "Original RealSense Data/frame_0011_depth.npy"
    CAD_SCALE_FACTOR = 0.00168095  # Your unified assembly scaling factor for non-removed gearbox STL model

    realsense_depth = np.load(NUMPY_SCAN_PATH)
    depth_image = o3d.geometry.Image(realsense_depth)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 615.0, 615.0, 320.0, 240.0)
    
    raw_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_image, intrinsic=intrinsics, depth_scale=1000.0, depth_trunc=3.0)
    
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)
    raw_target_cad = mesh.sample_points_uniformly(number_of_points=20000)
    
    print(f"[RAW DATA POINT COUNTS]")
    print(f"-> Raw Realsense Scan Points: {len(raw_pcd.points)}")
    print(f"-> Raw CAD Model Points:       {len(raw_target_cad.points)}\n")

    # 2. Track Crop and Plane Segmentation Effects
    min_bound = np.array([-1, -1, -1])
    max_bound = np.array([1, 1, 0.7])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_source = raw_pcd.crop(bbox)
    print(f"[PROCESSING STAGE 1: CROP]")
    print(f"-> Points remaining after workspace crop: {len(cropped_source.points)}")

    plane_model, inliers = cropped_source.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=200)
    segmented_source = cropped_source.select_by_index(inliers, invert=True)
    clean_source, _ = segmented_source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    print(f"[PROCESSING STAGE 2: PLANE REMOVAL]")
    print(f"-> Points remaining after removing floor/table: {len(clean_source.points)}")
    if len(clean_source.points) < 500:
        print("⚠️ WARNING: Plane segmentation might be accidentally deleting your gearbox!")

    # 3. Check Scale and Physical Bounding Boxes
    print(f"\n[PROCESSING STAGE 3: SCALE & BOUNDING BOXES]")
    src_box = clean_source.get_axis_aligned_bounding_box()
    src_extent = src_box.get_extent()
    print(f"-> Scan Physical Size (X, Y, Z): {src_extent[0]*100:.2f}cm x {src_extent[1]*100:.2f}cm x {src_extent[2]*100:.2f}cm")

    # Let's see what happens to the CAD scale
    raw_cad_extent = raw_target_cad.get_axis_aligned_bounding_box().get_extent()
    print(f"-> CAD Unscaled Size (X, Y, Z):  {raw_cad_extent[0]*100:.2f}cm x {raw_cad_extent[1]*100:.2f}cm x {raw_cad_extent[2]*100:.2f}cm")
    
    # Apply your script's scaling factor
    scaled_target_cad = copy.deepcopy(raw_target_cad)
    scaled_target_cad.scale(CAD_SCALE_FACTOR, center=scaled_target_cad.get_center())
    cad_extent = scaled_target_cad.get_axis_aligned_bounding_box().get_extent()
    print(f"-> CAD Scaled Size (X, Y, Z):    {cad_extent[0]*100:.2f}cm x {cad_extent[1]*100:.2f}cm x {cad_extent[2]*100:.2f}cm")

    # 4. Check for Downsampling Sufficiency
    print(f"\n[PROCESSING STAGE 4: DOWNSAMPLING EFFECT]")
    for v in [0.001, 0.002, 0.004]:
        ds_src = clean_source.voxel_down_sample(v)
        print(f"-> At Voxel Size {v*1000:.1f}mm, remaining source points = {len(ds_src.points)}")

    print("\n==================================================")

if __name__ == "__main__":
    import copy
    run_diagnostics()