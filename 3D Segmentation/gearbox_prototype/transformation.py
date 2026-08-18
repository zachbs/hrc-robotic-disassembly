import numpy as np
from scipy.spatial.transform import Rotation as R
import math

# Load static calibration & offset matrices once when module is imported
T_FLANGE_CAM = np.load("matrix_config/T_flange_cam.npy")
T_OBJ_GRASP = np.load("matrix_config/T_obj_grasp.npy")


def pose_to_matrix(pose):
    """Converts UR pose vector [x, y, z, rx, ry, rz] to 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T

def matrix_to_pose(T):
    """Converts 4x4 matrix to UR pose vector [x, y, z, rx, ry, rz]."""
    pos = T[:3, 3]
    rot_vec = R.from_matrix(T[:3, :3]).as_rotvec()
    return list(pos) + list(rot_vec)

def compute_base_to_cam(T_base_flange, T_flange_cam=T_FLANGE_CAM):
    """Computes the transformation from base to camera frame."""
    return T_base_flange @ T_flange_cam

def compute_base_to_obj(T_base_cam, T_cam_obj):
    """Computes the transformation from base to object frame."""
    return T_base_cam @ T_cam_obj

def compute_base_to_grasp(T_base_obj, T_obj_grasp=T_OBJ_GRASP):
    """Computes the transformation from base to grasp frame."""
    return T_base_obj @ T_obj_grasp

def compute_base_to_flange(rtde_r):
    """Computes the transformation from base to flange frame."""
    # This function would typically require the current pose of the robot's flange.
    # For demonstration, we will assume a placeholder identity matrix.
    raw_pose = rtde_r.getActualTCPPose()
    return pose_to_matrix(raw_pose)

def return_base_to_grasp_pose(rtde_r, T_cam_obj,T_flange_cam=T_FLANGE_CAM, T_obj_grasp=T_OBJ_GRASP):
    """Returns a function that computes the transformation from base to grasp frame."""
    print("T_flange_cam:", T_flange_cam)
    print("T_cam_obj:", T_cam_obj)
    T_base_flange = compute_base_to_flange(rtde_r)
    print("T_base_flange:", T_base_flange)
    T_base_cam = compute_base_to_cam(T_base_flange, T_flange_cam)
    print("T_base_cam:", T_base_cam)
    T_base_obj = compute_base_to_obj(T_base_cam, T_cam_obj)
    print("T_base_obj:", T_base_obj)
    T_base_grasp = compute_base_to_grasp(T_base_obj, T_obj_grasp)
    print("T_base_grasp:", T_base_grasp)
    final_pose = matrix_to_pose(T_base_grasp)
    final_pose[2] += 0.15  # Adjust Z-coordinate as needed
    print("Final pose to return:", final_pose)
    return final_pose


def get_all_poses(T_cam_obj,rtde_r):
    """Returns a dictionary of all candidate poses for grasping."""
    T_base_flange = compute_base_to_flange(rtde_r)
    print("T_base_flange:", T_base_flange)
    T_base_cam = compute_base_to_cam(T_base_flange)
    print("T_base_cam:", T_base_cam)
    T_base_obj = compute_base_to_obj(T_base_cam, T_cam_obj)
    print("T_base_obj:", T_base_obj)
    T_base_grasp = compute_base_to_grasp(T_base_obj)
    print("T_base_grasp:", T_base_grasp)


def get_all_poses_given_flange(T_cam_obj,T_base_flange):
    """Returns a dictionary of all candidate poses for grasping."""
    T_base_cam = compute_base_to_cam(T_base_flange)
    print("T_base_cam:", T_base_cam)
    T_base_obj = compute_base_to_obj(T_base_cam, T_cam_obj)
    print("T_base_obj:", T_base_obj)
    T_base_grasp = compute_base_to_grasp(T_base_obj)
    print("T_base_grasp:", T_base_grasp)
    
    return T_base_cam, T_base_obj, T_base_grasp






# 1. ALLOWED JOINT LIMITS
MIN_LIMITS_DEG = [-360, -360, -360, -210, -210, -360]
MAX_LIMITS_DEG = [ 360,  360,  360,  210,  210,  360]

MIN_LIMITS_RAD = [math.radians(deg) for deg in MIN_LIMITS_DEG]
MAX_LIMITS_RAD = [math.radians(deg) for deg in MAX_LIMITS_DEG]

# 2. ROTATION FLIPS
R_X_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
R_Y_180 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
R_Z_180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def wrap_to_pi(angle_rad):
    """Wraps an angle in radians to [-pi, pi]."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def normalize_wrist_joints(q_solution):
    """Safely normalizes wrist angles if a valid 6-DOF solution exists."""
    # Guard against empty list [] or None returned by failed IK solver
    if not q_solution or len(q_solution) != 6:
        return None
        
    q_norm = list(q_solution)
    for i in range(3, 6):
        q_norm[i] = wrap_to_pi(q_norm[i])
    return q_norm


def get_candidate_poses(T_base_grasp):
    return {
        "Original Pose": T_base_grasp,
        "180° Z-Flip (Yaw)": T_base_grasp @ R_Z_180,
        "180° X-Flip (Pitch)": T_base_grasp @ R_X_180,
        "180° Y-Flip (Roll)": T_base_grasp @ R_Y_180
    }


def is_joint_solution_safe(q_solution, min_limits=MIN_LIMITS_RAD, max_limits=MAX_LIMITS_RAD):
    if q_solution is None or len(q_solution) != 6:
        return False
        
    for i in range(6):
        if not (min_limits[i] <= q_solution[i] <= max_limits[i]):
            deg_val = math.degrees(q_solution[i])
            print(f"    -> Limit Breach: Joint {i+1} = {deg_val:.1f}°")
            return False
            
    return True


def execute_safe_grasp(rtde_c, rtde_r, T_base_grasp):
    candidate_poses = get_candidate_poses(T_base_grasp)
    q_current = rtde_r.getActualQ()
    valid_solutions = []
    
    print("\n--- Testing Orientation Candidates ---")
    for name, T_cand in candidate_poses.items():
        pose_array = matrix_to_pose(T_cand)
        
        # Check target distance before calling IK
        target_dist = np.linalg.norm(pose_array[:3])
        if target_dist > 0.85:
            print(f"\n[✗] {name}: Target is {target_dist:.3f}m from base (Exceeds UR reach envelope ~0.85m)!")
            continue

        # Solve IK
        q_sol = rtde_c.getInverseKinematics(pose_array, qnear=q_current)
        
        # Normalize wrist angles safely (returns None if q_sol is empty [])
        q_sol = normalize_wrist_joints(q_sol)
        
        if q_sol is None:
            print(f"  [✗] {name}: No IK solution found (Unreachable or singular pose)")
            continue

        if is_joint_solution_safe(q_sol):
            joint_dist = np.linalg.norm(np.array(q_sol) - np.array(q_current))
            valid_solutions.append((joint_dist, name, q_sol))
            print(f"  [✓] {name}: SAFE (Joint Travel: {math.degrees(joint_dist):.1f}°)")
        else:
            print(f"  [✗] {name}: REJECTED (Exceeds safety limits)")
            
    if not valid_solutions:
        raise RuntimeError(
            "CRITICAL: All orientation candidates failed. "
            "Check your +30cm Z approach offset—it is pushing the target past the robot's physical reach."
        )
        
    valid_solutions.sort(key=lambda item: item[0])
    best_dist, best_name, best_q = valid_solutions[0]
    
    print(f"\n[SUCCESS] Selected '{best_name}' as optimal grasp pose.")
    rtde_c.moveJ(best_q, speed=0.4, acceleration=0.4)





    