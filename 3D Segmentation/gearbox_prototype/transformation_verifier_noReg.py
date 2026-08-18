import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# 1. Load your original CAD point cloud
cad_pcd = o3d.io.read_point_cloud("optimized_cad_target.pcd")

# 2. Check where the center currently sits relative to (0,0,0)
old_center = cad_pcd.get_center()
print(f"[*] Current CAD center relative to origin: {old_center}")

# 3. Shift the CAD points so the geometric center becomes EXACTLY (0,0,0)
cad_pcd.translate(-old_center)

# 4. Save the pristine, centered CAD file
output_path = "centered_cad_target.pcd"
o3d.io.write_point_cloud(output_path, cad_pcd)
print(f"[+] Successfully saved centered CAD model to '{output_path}'!")
print(f"[*] New CAD center: {cad_pcd.get_center()}")

# ==============================================================================
# PRE-RECORDED MATRICES (FROM DOCUMENT)
# ==============================================================================

# Base to Flange (T_base_flange)
T_BASE_FLANGE = np.array([
    [ 0.9999,     0.013,      0.0058368,  0.28511],
    [ 0.013451,  -0.72577,   -0.68781,   -0.3006 ],
    [-0.0047052,  0.68781,   -0.72587,    0.3203 ],
    [ 0.0,        0.0,        0.0,        1.0    ]
])


# Sample Camera-to-Object transformation runs (T_cam_obj)
T_CAM_OBJ_SAMPLES = [
    np.array([
        [-0.64582,   0.55005,   0.52949,  -0.18829],
        [-0.76328,  -0.48149,  -0.43078,   0.23153],
        [ 0.017994, -0.68236,   0.7308,    0.10569],
        [ 0.0,       0.0,       0.0,       1.0    ]
    ]),
    np.array([
        [-0.63436,   0.55555,   0.53754,  -0.19356],
        [-0.77272,  -0.4755,   -0.42047,   0.22824],
        [ 0.022011, -0.6821,    0.73093,   0.10552],
        [ 0.0,       0.0,       0.0,       1.0    ]
    ]),
    np.array([
        [-0.63953,   0.55386,   0.53314,  -0.19041],
        [ 0.76846,   0.48018,   0.42297,  -0.15601],
        [-0.021736,  0.68019,  -0.73271,   0.62021],
        [ 0.0,       0.0,       0.0,       1.0    ]
    ]),
    np.array([
        [-0.63467,   0.54826,   0.54462,  -0.19359],
        [-0.77274,  -0.4575,   -0.43996,   0.2339 ],
        [ 0.0079544,-0.70007,   0.71403,   0.11259],
        [ 0.0,       0.0,       0.0,       1.0    ]
    ])
]




def apply_180_flip(T, axis='x'):
    """
    Flips a 4x4 transformation matrix 180 degrees around a specified local axis.
    
    Parameters:
        T (np.ndarray): 4x4 transformation matrix (e.g., T_cam_obj).
        axis (str): Local axis to flip around ('x', 'y', or 'z').
        
    Returns:
        np.ndarray: Flipped 4x4 transformation matrix.
    """
    axis = axis.lower()
    
    # 180-degree rotation matrices in 4x4 homogeneous coordinates
    flips = {
        'x': np.array([
            [ 1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0, -1, 0],
            [ 0,  0,  0, 1]
        ]),
        'y': np.array([
            [-1,  0,  0, 0],
            [ 0,  1,  0, 0],
            [ 0,  0, -1, 0],
            [ 0,  0,  0, 1]
        ]),
        'z': np.array([
            [-1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1]
        ])
    }
    
    if axis not in flips:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")
        
    # Post-multiplication applies the rotation in the local frame
    return T @ flips[axis]

# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================
def visualize_transforms(transforms_dict, axis_length=0.15):
    """Plots 4x4 transformation matrices as 3D coordinate frames."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(0, 0, 0, color="k", s=50, label="Origin (0,0,0)")

    for name, T in transforms_dict.items():
        origin = T[:3, 3]
        x_axis = T[:3, 0] * axis_length
        y_axis = T[:3, 1] * axis_length
        z_axis = T[:3, 2] * axis_length

        ax.quiver(*origin, *x_axis, color="r", linewidth=2)
        ax.quiver(*origin, *y_axis, color="g", linewidth=2)
        ax.quiver(*origin, *z_axis, color="b", linewidth=2)
        ax.text(origin[0], origin[1], origin[2] + 0.03, name, fontsize=10, weight="bold")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Transformation Verifier (Red=X, Green=Y, Blue=Z)")

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    centers = np.mean(limits, axis=1)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])
    plt.show()

# ==============================================================================
# MAIN VERIFICATION PIPELINE
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("      STATIC TRANSFORMATION MATRIX VERIFIER")
    print("=" * 60)
    import transformation  # Ensure transformation.py is available
    # Use primary sample run for T_cam_obj
    T_cam_obj = np.linalg.inv(T_CAM_OBJ_SAMPLES[0])
    # T_cam_obj = apply_180_flip(T_cam_obj, axis='x')  # Apply 180-degree flip around X-axis
    
    T_base_cam, T_base_obj, T_base_grasp = transformation.get_all_poses_given_flange(T_cam_obj, T_BASE_FLANGE)

    print("\n[+] T_base_flange:\n", T_BASE_FLANGE)
    print("\n[+] T_base_cam:\n", T_base_cam)
    print("\n[+] Selected T_cam_obj (Sample 1):\n", T_cam_obj)
    print("\n[+] Calculated T_base_obj (T_base_cam @ T_cam_obj):\n", T_base_obj)
    print("\n[+] Recorded T_base_obj Reference:\n", T_base_obj)
    print("\n[+] Recorded T_base_grasp Reference:\n", T_base_grasp)

    # Try calling transformation.py if available without live RTDE
    try:
        import transformation
        if hasattr(transformation, "get_all_poses"):
            print("\n[*] 'transformation.py' module detected.")
            # Note: Update function signature if transformation.py requires custom inputs
    except ImportError:
        print("\n[*] 'transformation.py' not imported; running in standalone verifier mode.")

    # Frame dictionary for 3D visualization
    transforms_dict = {
        "Base (0,0,0)": np.eye(4),
        "Flange": T_BASE_FLANGE,
        "Camera": T_base_cam,
        "Object (Calculated)": T_base_obj,
        "Object (Recorded)": T_base_obj,
        "Grasp Target": T_base_grasp,
    }

    print("\n[*] Launching 3D Coordinate Frame Visualizer...")
    visualize_transforms(transforms_dict)

if __name__ == "__main__":
    main()