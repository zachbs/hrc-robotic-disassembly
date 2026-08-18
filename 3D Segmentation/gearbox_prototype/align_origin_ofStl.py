import os
import open3d as o3d


def align_and_center_clouds(
    source_path: str, target_path: str, output_path: str, visualize: bool = True
):
    """Translates a target PCD/mesh to match the centroid of a source PCD scan

    and saves the result.
    """
    # 1. Load the Point Clouds / Models
    print(f"[INFO] Loading source: {source_path}")
    source = o3d.io.read_point_cloud(source_path)

    print(f"[INFO] Loading target: {target_path}")
    # Supports both .pcd and 3D mesh files (.stl, .ply, .obj)
    target = o3d.io.read_point_cloud(target_path)

    if source.is_empty() or target.is_empty():
        raise FileNotFoundError(
            "Could not load one or both point clouds. Check file paths."
        )

    


    # 2. Check where the center currently sits relative to (0,0,0)
    old_center = target.get_center()
    print(f"[*] Current target center relative to origin: {old_center}")

    # 3. Shift the target points so the geometric center becomes EXACTLY (0,0,0)
    target.translate(-old_center)


    # Optional: Visualize BEFORE alignment
    if visualize:
        source.paint_uniform_color([0.8, 0.1, 0.1])  # Red = Source Scan
        target.paint_uniform_color([0.1, 0.8, 0.1])  # Green = Unaligned Target
        print("[INFO] Displaying UNALIGNED clouds (Close window to proceed)...")
        o3d.visualization.draw_geometries(
            [source, target], window_name="Before Centroid Translation"
        )


    print(f"-> Translation Vector Applied    : {-old_center}")
    print(f"-> New Target Center             : {target.get_center()}")

    # Optional: Visualize AFTER alignment
    if visualize:
        target.paint_uniform_color([0.1, 0.4, 0.9])  # Blue = Aligned Target
        print("[INFO] Displaying ALIGNED clouds...")
        o3d.visualization.draw_geometries(
            [source, target], window_name="After Centroid Translation"
        )

    # 4. Save the Centered/Translated Target
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    success = o3d.io.write_point_cloud(output_path, target)

    if success:
        print(
            f"[SUCCESS] Centered target saved successfully to: {output_path}"
        )
    else:
        print(f"[ERROR] Failed to save output to: {output_path}")


if __name__ == "__main__":
    # --- Configuration Paths ---
    SOURCE_PCD = "captured_scans/gearbox_scan_20260616_172713.pcd"  # Path to your scene/camera scan
    TARGET_PCD = "captured_scans/optimized_cad_target2.pcd"  # Path to your CAD model / template
    OUTPUT_PCD = "captured_scans/origin_centered_local_global.pcd"  # Output file location

    align_and_center_clouds(
        source_path=SOURCE_PCD,
        target_path=TARGET_PCD,
        output_path=OUTPUT_PCD,
        visualize=True,  # Set to False for headless execution
    )