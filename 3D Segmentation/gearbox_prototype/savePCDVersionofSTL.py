import os
import numpy as np
import open3d as o3d

# ==============================================================================
# CONFIGURATION
# ==============================================================================
STL_FILE_PATH = "nonScaledFullGearboxInsideRemoved-Fusion_CenteredOrigin.stl"
OUTPUT_DIR = "captured_scans"
OUTPUT_FILENAME = "optimized_cad_target2.pcd"

# Sampling & Scaling parameters (From your original pipeline)
NUMBER_OF_POINTS = 30000
SCALE_FACTOR = 0.00168095
VOXEL_SIZE = 0.0025

# ==============================================================================
# NORMAL VECTOR TUNING PARAMETERS
# ==============================================================================
# Toggle to re-calculate normals on point cloud vs using STL mesh face normals
ESTIMATE_FRESH_NORMALS = True
NORMAL_SEARCH_RADIUS = VOXEL_SIZE * 2  # 0.005
NORMAL_MAX_NN = 30

# Set to True if normals are pointing inside/backwards and need to be inverted
FLIP_NORMALS = False

# Optional: Orient all normals towards a specific reference coordinate
ORIENT_TOWARDS_POINT = False
CAMERA_LOCATION = np.array([0.0, 0.0, 0.0])


def main():
    # 1. Ensure output storage directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[+] Created directory: '{OUTPUT_DIR}'")

    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # 2. Load STL Mesh
    print(f"\n[*] Loading STL mesh: {STL_FILE_PATH}")
    mesh = o3d.io.read_triangle_mesh(STL_FILE_PATH)

    if mesh.is_empty():
        print(f"[-] CRITICAL ERROR: Could not read STL file at '{STL_FILE_PATH}'")
        return

    # Pre-compute mesh face/vertex normals
    mesh.compute_vertex_normals()

    # 3. Uniformly sample points from the mesh geometry
    print(f"[*] Sampling {NUMBER_OF_POINTS} surface points...")
    pcd = mesh.sample_points_uniformly(number_of_points=NUMBER_OF_POINTS)

    # 4. Scale & Downsample (Matching your registration setup)
    if SCALE_FACTOR != 1.0:
        print(f"[*] Applying scale factor ({SCALE_FACTOR})...")
        pcd.scale(SCALE_FACTOR, center=pcd.get_center())

    if VOXEL_SIZE > 0:
        print(f"[*] Voxel downsampling (Voxel Size = {VOXEL_SIZE})...")
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    # 5. Generate and Adjust Normals
    if ESTIMATE_FRESH_NORMALS:
        print(
            f"[*] Re-estimating point cloud normals (Radius={NORMAL_SEARCH_RADIUS}, MaxNN={NORMAL_MAX_NN})..."
        )
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_SEARCH_RADIUS, max_nn=NORMAL_MAX_NN
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    if ORIENT_TOWARDS_POINT:
        print(f"[*] Orienting normals towards point: {CAMERA_LOCATION}...")
        pcd.orient_normals_towards_camera_location(
            camera_location=CAMERA_LOCATION
        )

    if FLIP_NORMALS:
        print("[*] Inverting all normal vectors 180 degrees...")
        pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

    # 6. Overwrite the PCD File
    print(f"[*] Writing PCD file to disk...")
    success = o3d.io.write_point_cloud(output_filepath, pcd)

    if success:
        print(
            f"[+] SUCCESS: Overwritten '{output_filepath}' ({len(pcd.points)} points with normals)."
        )
    else:
        print(f"[-] ERROR: Failed to write PCD file to '{output_filepath}'")

    # 7. Interactive Visual Verification
    print("\n[*] Opening Open3D Visualizer...")
    print("    -> Press '+' / '-' in the window to increase/decrease normal line lengths.")
    print("    -> Press 'N' to toggle normal vector displays on or off.")

    pcd.paint_uniform_color([0, 0.651, 0.929])  # Cyan
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="PCD Normal Verification (Close to Exit)",
        point_show_normal=True,
        width=1024,
        height=768,
    )


if __name__ == "__main__":
    main()