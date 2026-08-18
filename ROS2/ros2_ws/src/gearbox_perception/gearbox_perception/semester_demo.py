# Demo showcasing different areas of this semester
# Mostly skeleton code for now 
from loop_pickup_with_adaptive_gripper import *
from gui_ai import *
from ultralytics import YOLO
from intent_engine_v3 import IntentEngine, EngineConfig
import gearbox_3D_Full_Pipeline



    
# stop callback function
def button_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if HALF_WIDTH + 50 <= x <= HALF_WIDTH + 250 and HALF_HEIGHT + 50 <= y <= HALF_HEIGHT + 120:
            add_log("Start Button clicked!")
            params["emergency_stop"].clear()
        # STOP button
        elif HALF_WIDTH + 300 <= x <= HALF_WIDTH + 500 and HALF_HEIGHT + 50 <= y <= HALF_HEIGHT + 120:
            add_log("Stop Button clicked!")
            params["emergency_stop"].set()
        # RESET button
        elif HALF_WIDTH + 50 <= x <= HALF_WIDTH + 500 and HALF_HEIGHT + 150 <= y <= HALF_HEIGHT + 220:
            add_log("Reset Button clicked!")
            with params["state_lock"]:
                rtde_control.stopJ(2.0)  # stop robot motion immediately
                params["shared_state"]["robot_state"] = 1
                params["shared_state"]["robot_moving"] = "IDLE"




def human_detection_loop(pipeline, align, shared_frame,
                   frame_lock, stop_event, emergency_stop_event, intent_engine):
    while not stop_event.is_set():
        # first realsense, the one mounted above
        overview_frames = pipeline.wait_for_frames()
        overview_aligned = align.process(overview_frames)
        overview_color = overview_aligned.get_color_frame()
        if not overview_color:
            continue
            
        overview_img = np.asanyarray(overview_color.get_data())

        # Push frame to the intent engine 
        if intent_engine is not None:
            intent_engine.push_frame(overview_img)
            
            # Optional: Draw current state for the GUI
            state = intent_engine.get_state()
            cv2.putText(overview_img, f"Intent: {state['label']} ({state['confidence']:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        with frame_lock:
            shared_frame["overview"] = overview_img.copy()


def realsense_loop(shared_state, pipeline, align, shared_frame, shared_detections,
                   frame_lock, detection_lock, stop_event,state_lock):

    while not stop_event.is_set():
        # second realsense, mounted on ur5e
        robot_frames = pipeline.wait_for_frames()
        aligned = align.process(robot_frames)
        color = aligned.get_color_frame()

        if not color:
            continue

        img = np.asanyarray(color.get_data())
        vis = img.copy()

        detections = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)

            if area < 5000 or area > 100000:
                continue

            cv2.drawContours(vis, contours, i, (0, 0, 255), 2)

            try:
                angle, center_point = getOrientation(c, vis)
                detections.append((c, area, angle, center_point))
            except Exception as e:
                print(f"[WARN] PCA failed: {e}")
                continue

        if shared_state["robot_state"] == 2 or shared_state["robot_state"] == 5 and shared_state["robot_moving"] == "DETECTING FASTENERS":
            
            results = model(img, conf=0.75, verbose=False)
            if results is None or len(results) == 0:
                print("[WARN] No detections from YOLO model")
            # 5. Process Detections
            for r in results:
                for box in r.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    

                    # Draw info
                    cls_name = model.names[int(box.cls[0])]
                    label = f"{cls_name}"

                    if "gearbox" in cls_name.lower():
                        # Save the center pixel for the robot loop
                        with state_lock:
                            shared_state["gearbox_center"] = (cx, cy)

                    if "fastener" in cls_name.lower():
                        shared_state["fasteners_detected"] += 1
                    
                    color = (255, 255, 255) if "gearbox" in cls_name.lower() else (0, 255, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if shared_state["robot_state"] == 3 and shared_state["robot_moving"] == "MOVING":
            try:
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                    
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                color_image = np.asanyarray(color_frame.get_data()).copy()
                
                # 1. Run inference to keep updating the live preview bounding box
                results = model(color_image, conf=0.60, verbose=False)[0]
                bbox = results.boxes[0].xyxy[0].cpu().numpy().astype(int) if len(results.boxes) > 0 else None
                
                # Only update the tracking box if YOLO actively sees it
                if bbox is not None:
                    with state_lock:
                        shared_state["captured_frame_bbox"] = bbox

                # 2. Overwrite 'vis' for UI rendering
                vis = results.plot()

                # Get thread-safe states
                with state_lock:
                    is_capturing = shared_state.get("capturing_frames", False)
                with frame_lock:
                    current_buffer_len = len(shared_frame["captured_frame_buffer"])

                
                # 3. Draw text feedback status on the UI frame
                if not is_capturing:
                    cv2.putText(vis, "READY: PRESS 's' ON MAIN WINDOW", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, f"CAPTURING FRAMES: {current_buffer_len}/{FRAMES_TO_CAPTURE}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 4. ROBUST CAPTURE LOGIC: Fill the buffer safely without frame-drop cancellations
                if is_capturing:
                    if current_buffer_len < FRAMES_TO_CAPTURE:
                        with frame_lock:
                            shared_frame["captured_frame_buffer"].append(depth_image)
                    else:
                        print("[+] Target Buffer Locked. Burst capture complete.")
                        with state_lock:
                            shared_state["capturing_frames"] = False
                            
                            
            except Exception as e:
                print(f"[WARN] Error in live preview: {e}")

        # --- update shared data ---
        with frame_lock:
            shared_frame["ur5e"] = vis.copy()
            shared_frame["ur5e_depth"] = aligned.get_depth_frame()

        # Sort to try to match with gearbox area
        target_area = 19600
        detections.sort(key=lambda d: abs(d[1] - target_area))

        with detection_lock:
            shared_detections["objects"] = detections
            # if detections:
            #     print(detections[0][1])

        

        
def is_stopped():
    qd = rtde_receive.getActualQd()
    return max(abs(v) for v in qd) < 0.01

def robot_loop(shared_state, shared_detections, detection_lock, stop_event, emergency_stop_event, pipeline, shared_frame, state_lock, frame_lock):
    def get_offsets(current_pose, offset_magnitude, offset_angle):
        # Convert to pitch roll yaw
        axis_angle = np.array(current_pose[3:])
        # Convert axis-angle to rotation matrix
        r = R.from_rotvec(axis_angle)
        rotation_matrix = r.as_matrix()
        euler = r.as_euler('xyz', degrees=True)
        # Calculate offset for gripper based on tool's current orientation
        offset_x = offset_magnitude*cos(radians(euler[2])+offset_angle)
        offset_y = offset_magnitude*sin(radians(euler[2])+offset_angle)
        return offset_x, offset_y
    while not stop_event.is_set():  # your normal stop event
        if emergency_stop_event.is_set() and not is_stopped():
            add_log("[EMERGENCY STOP] Halting UR5e immediately!")
            try:
                rtde_control.stopJ(2.0)  # stop robot motion immediately
                shared_state["robot_moving"] = "IDLE"
            except Exception as e:
                print(f"[WARN] Could not stop UR5e: {e}")
                break  # exit loop
        elif not emergency_stop_event.is_set():
            if shared_state["robot_state"] == 1: # move to observe pose 
                try:
                    if shared_state["robot_moving"] == "IDLE" :   
                      add_log("Moving to observe pose")
                      gripper.open()
                      rtde_control.moveL(OBSERVE_POSE, 0.25, 0.25, True) 
                      shared_state["robot_moving"] = "MOVING"
                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED"
                        add_log("Arrived")
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Reached observe pose")
                        shared_state["robot_state"] = 2
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not reach observe pose: {e}")
            if shared_state["robot_state"] == 2: # Find Gearbox with YOLO & Move 8cm Above
                try:
                    if shared_state["robot_moving"] == "IDLE":  
                        
                        
                        # Grab the center pixel from YOLO (written by your vision thread)
                        with state_lock:
                            cx, cy = shared_state.get("gearbox_center", (None, None))
                        
                        if cx is not None:
                            add_log("Calculating YOLO Gearbox 3D Position")
                            # 1. Properly initialize Camera Intrinsics (K, D) and Transforms
                            profile = pipeline.get_active_profile()
                            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                            intrinsics = color_profile.get_intrinsics()
                            
                            fx = float(intrinsics.fx)
                            fy = float(intrinsics.fy)
                            ppx = float(intrinsics.ppx)
                            ppy = float(intrinsics.ppy)

                            # Build camera matrix K
                            K = np.array([[fx, 0.0, ppx],
                                        [0.0, fy, ppy],
                                        [0.0, 0.0, 1.0]], dtype=float)

                            # Load distortion and transformation matrices
                            D = np.array(intrinsics.coeffs[:5], dtype=float)
                            T_tcp_cam = np.load(T_TCP_CAM_PATH)
                            
                            with frame_lock:
                                ui = shared_frame["ur5e"].copy() if shared_frame["ur5e"] is not None else None

                            # 2. Convert YOLO pixel coordinates to Robot Base Coordinates
                            pose6 = get_tcp_pose6(rtde_receive)
                            T_base_tcp = ur_pose6_to_T(pose6)
                            
                            Xb, Yb, _, info = intersect_Z0(cx, cy, K, D, T_base_tcp, T_tcp_cam, ui, True, True)
                            
                            if Xb is not None:
                                # 3. Target Pose: 8cm above the object
                                # Keep current rotation vectors pointing straight down
                                Xb_corrected = Xb - 0.030 
                                Yb_corrected = Yb + 0.000 
                                target_pose = [Xb_corrected, Yb_corrected, 0.360, pose6[3], pose6[4], pose6[5]]
                                
                                # Store targets in shared state for subsequent states to access
                                with state_lock:
                                    shared_state["target_Xb"] = Xb_corrected
                                    shared_state["target_Yb"] = Yb_corrected
                                    shared_state["robot_moving"] = "MOVING"
                                
                                rtde_control.moveL(target_pose, 0.25, 0.25, True)
                                add_log(f"Moving above YOLO target: X={Xb:.3f}, Y={Yb:.3f}")
                        else:
                            add_log("Waiting for YOLO to detect gearbox...")
                            time.sleep(0.5) # Prevent log flooding if not seen yet
                                
                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        add_log("Hovering 8cm above Gearbox")
                        with state_lock:
                            shared_state["robot_state"] = 3
                            shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"Error in State 2: {e}")

            if shared_state["robot_state"] == 3:
                try:
                    if shared_state["robot_moving"] == "IDLE":
                        add_log("Orbiting to 45-degree scan angle")
                        Xb = shared_state["target_Xb"]
                        Yb = shared_state["target_Yb"]
                        orbit_pose = [Xb, Yb + 0.30, 0.33, 2.356, 0.0, 0.0]
                        
                        rtde_control.moveL(orbit_pose, 0.1, 0.1, True)
                        shared_state["robot_moving"] = "MOVING"
                        
                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        
                        
                        # Safe Isolation: Quick read access under lock protection
                        with state_lock:
                            buffer_ready = len(shared_frame["captured_frame_buffer"]) == 25
                            local_buffer = list(shared_frame["captured_frame_buffer"]) if buffer_ready else None
                            local_bbox = shared_state["captured_frame_bbox"].copy() if (buffer_ready and shared_state["captured_frame_bbox"] is not None) else None
                        if buffer_ready and local_buffer is not None:
                            # CRITICAL: Compute OUTSIDE the state_lock context to avoid cross-thread deadlocks
                            target, source, computed_pose = gearbox_3D_Full_Pipeline.return6DPose(local_buffer, local_bbox) 
                            # Re-engage lock only for a low-overhead memory write
                            with state_lock:
                                shared_state["6D_pose"] = computed_pose
                                shared_state["source_pc"] = source
                                shared_state["target_pc"] = target
                        else:
                            add_log("Triggering 3D Registration Pipeline")  

                        if shared_state["6D_pose"] is not None:
                            add_log(f"6D Pose Computed successfully. ")
                            time.sleep(1) # allow time for visualization and verification
                            with state_lock:
                                shared_state["robot_state"] = 5
                                shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"Error in State 3: {e}")

            if shared_state["robot_state"] == 4:
                # Grab Perpendicular and Flip Flat
                try:
                    if shared_state["robot_moving"] == "IDLE":
                        add_log("Executing Flip Maneuver")
                        
                        Xb = shared_state["target_Xb"]
                        Yb = shared_state["target_Yb"]
                        
                        # # STEP A: Move to side (Blocking check)
                        # side_grab_pose = [Xb, Yb - 0.08, 0.05, 1.2, -1.2, 1.2] 
                        # rtde_control.moveL(side_grab_pose, 0.2, 0.2, False) 
                        
                        # if stop_event.is_set() or emergency_stop_event.is_set(): continue

                        # # STEP B: Close Gripper
                        # gripper.close()
                        # time.sleep(1)
                        
                        # if stop_event.is_set() or emergency_stop_event.is_set(): continue
                        
                        # # STEP C: Lift up
                        # lift_pose = [Xb, Yb - 0.08, 0.200, 1.2, -1.2, 1.2]
                        # rtde_control.moveL(lift_pose, 0.2, 0.2, False)
                        
                        # if stop_event.is_set() or emergency_stop_event.is_set(): continue
                        
                        # # STEP D: Rotate wrist flat
                        # flat_pose = [Xb, Yb, 0.200, 3.1415, 0.0, 0.0]
                        # rtde_control.moveL(flat_pose, 0.2, 0.2, False)
                        
                        # if stop_event.is_set() or emergency_stop_event.is_set(): continue
                        
                        # # STEP E: Push down and release
                        # drop_pose = [Xb, Yb, 0.100, 3.1415, 0.0, 0.0]
                        # rtde_control.moveL(drop_pose, 0.1, 0.1, False)
                        # gripper.open()
                        time.sleep(1)
                        
                        shared_state["robot_moving"] = "MOVING"

                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        add_log("Gearbox oriented flat. Resetting to observe.")
                        rtde_control.moveL(OBSERVE_POSE, 0.25, 0.25, True)
                        
                        # Add this intermediate state instead of jumping to 5
                        shared_state["robot_moving"] = "RETURNING"
                        
                    # NEW BLOCK: Wait for the robot to actually stop at the observe pose
                    elif shared_state["robot_moving"] == "RETURNING" and is_stopped():
                        with state_lock:
                            shared_state["robot_state"] = 5
                            shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"Error in State 4: {e}")
                        
            if shared_state["robot_state"] == 5: # iterate to be above object
                try:
                    if shared_state["robot_moving"] == "IDLE":  
                        object_detected2 = False 
                        add_log("Detecting Objects")
                        with detection_lock:
                            current_detections = shared_detections["objects"].copy()
                        
                        if current_detections:
                            object_detected2 = True
                            total_distance = 100

                            # Get camera intrinsics
                            profile = pipeline.get_active_profile()
                            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                            intrinsics = color_profile.get_intrinsics()
                            
                            # Extract intrinsics parameters
                            fx = float(intrinsics.fx)
                            fy = float(intrinsics.fy)
                            cx = float(intrinsics.ppx)
                            cy = float(intrinsics.ppy)

                            # Build camera matrix K
                            K = np.array([[fx, 0.0, cx],
                                        [0.0, fy, cy],
                                        [0.0, 0.0, 1.0]], dtype=float)

                            T_tcp_cam = np.load(T_TCP_CAM_PATH)

                            ui = shared_frame["ur5e"]

                            # Get distortion coefficients
                            D = np.array(intrinsics.coeffs[:5], dtype=float)
                            if current_detections:
                                contour, area, angle, center_point = current_detections[0]
                                 # Get robot pose and compute intersection using center point
                                pose6 = get_tcp_pose6(rtde_receive)
                                T_base_tcp = ur_pose6_to_T(pose6)
                                
                                # Use center point from PCA as the "clicked point"
                                center_x, center_y = center_point
                                
                                # # Get position from center point using existing logic
                                # Xb, Yb, _, info = intersect_Z0(center_x, center_y, K, D, T_base_tcp, T_tcp_cam, ui, True, True)
                                
                                if Xb is not None:
                                    
                                    # Calculate aligned TCP pose for top-down camera
                                    current_pose = get_tcp_pose6(rtde_receive)
                                    current_x, current_y, current_z = current_pose[0:3]
                                    current_rx, current_ry, current_rz = current_pose[3:6]
                                    
                                    # For top-down camera: PCA angle directly maps to Rz (Yaw)
                                    # Keep current Rx, Ry unchanged, only adjust Rz
                                    aligned_rz = angle  # PCA angle in radians
                                    
                                    # Create aligned pose: [X, Y, Z, Rx, Ry, Rz]
                                    aligned_pose = [Xb, Yb, 0.32, 3.1415, current_ry, current_rz]

                                    # move to pose
                                    rtde_control.moveL(aligned_pose, 0.25, 0.25, True) 
                                    
                        if object_detected2:
                            shared_state["robot_moving"] = "MOVING"
                            add_log("Moving to aligned pose")        

                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED"
                        
                        # Calculate current distance from target
                        distance_X = abs(current_x - Xb)
                        distance_Y = abs(current_y - Yb)
                        total_distance = ((distance_X**2 + distance_Y**2)**0.5) * 1000  # Convert to mm
                        if total_distance > 10:
                            shared_state["robot_moving"] = "IDLE" # Trigger another iteration
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Reached first aligned pose")
                        time.sleep(0.5)
                        with frame_lock:
                            depth_frame = shared_frame["ur5e_depth"]
                        # Convert depth frame to numpy array
                        depth_image = np.asanyarray(depth_frame.get_data())
                        # Get depth scale
                        depth_scale = depth_frame.get_units()  # meters per depth unit
                        h, w = depth_image.shape
                        center_x, center_y = w // 2, h // 2 
                        depth_value = depth_image[center_y, center_x] * depth_scale # distance to top of object
                        print(depth_value)
                        current_pose = get_tcp_pose6(rtde_receive) # get current poses for next step
                        shared_state["robot_moving"] = "DETECTING FASTENERS"
                        shared_state["fasteners_detected"] = 0
                        add_log("Detecting fasteners")
                        time.sleep(6)
                    elif shared_state["robot_moving"] == "DETECTING FASTENERS":
                        
                        if shared_state["fasteners_detected"] != 0:
                            add_log("Fastener(s) detected! Resetting to observe pose")
                            time.sleep(4)
                            shared_state["robot_moving"] = "IDLE"
                            shared_state["robot_state"] = 5

                        else:
                            add_log("No fasteners detected. Removing lid")
                            shared_state["robot_state"] = 6
                            shared_state["robot_moving"] = "IDLE"

                except Exception as e:
                    print(f"[WARN] Could not reach aligned pose: {e}")
            if shared_state["robot_state"] == 6: # offset to center joint 5 above object
                try:
                    if shared_state["robot_moving"] == "IDLE" :   
                        add_log("Moving to offset pose")
                        offset_x, offset_y = get_offsets(current_pose,0.05,3*pi/4)
                        target_pose_step = [current_pose[0]+offset_x, current_pose[1]+offset_y, current_pose[2], current_pose[3], current_pose[4], current_pose[5]]  # Offset added
                        # Move
                        rtde_control.moveL(target_pose_step, 0.6, 0.6, True)

                        shared_state["robot_moving"] = "MOVING"
                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED"
                        add_log("Arrived")
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Reached offset pose")
                        shared_state["robot_state"] = 7
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not reach centered pose: {e}")
            if shared_state["robot_state"] == 7: # rotate to align
                try:
                    if shared_state["robot_moving"] == "IDLE" :   
                        add_log("Aligning angle")
                        with detection_lock:
                            current_detections = shared_detections["objects"].copy()
                        if current_detections:
                            contour, area, angle, center_point = current_detections[0]
                            # Get current joint angles
             
                        current_joints = rtde_receive.getActualQ()                                    
                        # Create target joint configuration with fresh angle data
                        target_joints = [
                            current_joints[0],  # current theta1
                            current_joints[1],  # current theta2
                            current_joints[2],  # current theta3
                            current_joints[3],  # current theta4
                            current_joints[4],         # theta5 = 90 degrees
                            current_joints[5] + angle  # current theta6 + fresh angle
                        ]
                        # Execute moveJ command
                        rtde_control.moveJ(target_joints, 1, 1, True)
                        shared_state["robot_moving"] = "ARRIVED"
                        add_log("Arrived")

                    elif shared_state["robot_moving"] == "MOVING":    # FIX THIS LATER
                        shared_state["robot_moving"] = "IDLE"
                        
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Reached aligned angle")
                        shared_state["robot_state"] = 8
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not reach rotated pose: {e}")
            if shared_state["robot_state"] == 8: # pick up lid
                try:
                    if shared_state["robot_moving"] == "IDLE" and is_stopped():   
                        add_log("Picking Up object")
                        current_pose = get_tcp_pose6(rtde_receive) # get current poses for next step
                        offset_x, offset_y = get_offsets(current_pose,-0.076,pi/2)
                        target_z = current_pose[2]-depth_value+0.1
                        print(target_z)
                        if target_z < 0.16:
                            target_z = 0.16
                        if target_z > 0.25:
                            target_z = 0.25
                        above_pose = [current_pose[0]+offset_x, current_pose[1]+offset_y, current_pose[2], current_pose[3], current_pose[4], current_pose[5]]
                        target_pose = [above_pose[0], above_pose[1], target_z, above_pose[3], above_pose[4], above_pose[5]]  # Offset added
                        # Move
                        rtde_control.moveL(target_pose, 0.6, 0.6, True)

                        shared_state["robot_moving"] = "MOVING"
                    elif shared_state["robot_moving"] == "MOVING" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED"
                        add_log("Arrived")
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Reached pick pose, grabbing")
                        gripper.close()
                        shared_state["robot_state"] = 9
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not pickup lid: {e}")
            if shared_state["robot_state"] == 9: # move lid to the side 
                try:
                    if shared_state["robot_moving"] == "IDLE" :   
                        add_log("Moving lid")
                        current_pose = get_tcp_pose6(rtde_receive) # get current pose
                        intermediate_pose = [current_pose[0],current_pose[1],current_pose[2]+0.09,current_pose[3],current_pose[4],current_pose[5]]
                        rtde_control.moveL(intermediate_pose, 0.25, 0.25, True) 
                        shared_state["robot_moving"] = "MOVING1"
                    elif shared_state["robot_moving"] == "MOVING1" and is_stopped():
                        shared_state["robot_moving"] = "MOVING2"
                        # rtde_control.moveL(OBSERVE_POSE, 0.25, 0.25, True) 
                    elif shared_state["robot_moving"] == "MOVING2" and is_stopped():
                        shared_state["robot_moving"] = "MOVING3"
                        place_pose = [0.52, -0.6, 0.152, 3.14159, 0.0, 0.0]
                        rtde_control.moveL(place_pose, 0.25, 0.25, True)
                    elif shared_state["robot_moving"] == "MOVING3" and is_stopped():
                        shared_state["robot_moving"] = "PLACING"
                        gripper.open()
                        # rtde_control.moveL(OBSERVE_POSE, 0.25, 0.25, True) 
                    elif shared_state["robot_moving"] == "PLACING" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED"
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Reached observe pose")
                        shared_state["robot_state"] = 10
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not move lid: {e}")
            if shared_state["robot_state"] == 10: # pick up gear 1
                try:
                    if shared_state["robot_moving"] == "IDLE" and is_stopped():   
                        add_log("Picking Up Gear 1")
                        target_pose2 = [target_pose[0],target_pose[1],target_pose[2]+0.04,target_pose[3],target_pose[4],target_pose[5]] # above pose from earlier step
                        rtde_control.moveL(target_pose2, 0.25, 0.25, True) 
                        shared_state["robot_moving"] = "MOVING1"
                    elif shared_state["robot_moving"] == "MOVING1" and is_stopped():
                        shared_state["robot_moving"] = "MOVING2" 
                        current_pose = target_pose2 # do not get these from RTDE receive so that pausing will not mess up 
                        offset_x, offset_y = get_offsets(current_pose, 0.03217,0.8) # calculated experimentally 0.03217  -0.892
                        gripper.move(140, 10,100) # position, speed, force
                        target_pose3 = [current_pose[0]+offset_x, current_pose[1]+offset_y, current_pose[2], current_pose[3], current_pose[4], current_pose[5]]  # Offset added
                        rtde_control.moveL(target_pose3, 0.6, 0.6, True) # Move
                    elif shared_state["robot_moving"] == "MOVING2" and is_stopped():
                        shared_state["robot_moving"] = "MOVING3" 
                        current_pose = get_tcp_pose6(rtde_receive) # get current poses
                        target_pose4 = [target_pose3[0], target_pose3[1], target_pose3[2]-0.017, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose4, 0.6, 0.6, True) # Move
                        gripper.close()
                    elif shared_state["robot_moving"] == "MOVING3" and is_stopped():
                        add_log("Gear 1 Grabbed")
                        target_pose5 = [target_pose3[0], target_pose3[1], target_pose3[2]+0.09  , target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose5, 0.6, 0.6, True) # Move
                        shared_state["robot_moving"] = "MOVING4" 
                    elif shared_state["robot_moving"] == "MOVING4" and is_stopped():
                        add_log("Placing Gear 1")
                        place_pose = [0.662, -0.378, 0.212, 3.14159, 0.0, 0.0] # Offset added
                        rtde_control.moveL(place_pose, 0.6, 0.6, True) # Move
                        shared_state["robot_moving"] = "MOVING5" 
                    elif shared_state["robot_moving"] == "MOVING5" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED" 
                        gripper.open()
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Gear 1 Removed")
                        shared_state["robot_state"] = 11
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not pickup first gear: {e}")
            if shared_state["robot_state"] == 11: # pick up gear2
                try:
                    if shared_state["robot_moving"] == "IDLE" and is_stopped():   
                        add_log("Picking Up Gear 2")
                        target_pose2 = [target_pose[0],target_pose[1],target_pose[2]+0.04,target_pose[3],target_pose[4],target_pose[5]] # above pose from earlier step
                        rtde_control.moveL(target_pose2, 0.25, 0.25, True) 
                        shared_state["robot_moving"] = "MOVING1"
                    elif shared_state["robot_moving"] == "MOVING1" and is_stopped():
                        shared_state["robot_moving"] = "MOVING2" 
                        current_pose = target_pose2 # do not get these from RTDE receive so that pausing will not mess up 
                        offset_x, offset_y = get_offsets(current_pose, -0.03217,0.8) # calculated experimentally 0.03217  -0.892
                        gripper.move(140, 10,100) # position, speed, force
                        target_pose3 = [current_pose[0]+offset_x, current_pose[1]+offset_y, current_pose[2], current_pose[3], current_pose[4], current_pose[5]]  # Offset added
                        rtde_control.moveL(target_pose3, 0.6, 0.6, True) # Move
                    elif shared_state["robot_moving"] == "MOVING2" and is_stopped():
                        shared_state["robot_moving"] = "MOVING3" 
                        current_pose = get_tcp_pose6(rtde_receive) # get current poses
                        target_pose4 = [target_pose3[0], target_pose3[1], target_pose3[2]-0.017, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose4, 0.6, 0.6, True) # Move
                        gripper.close()
                    elif shared_state["robot_moving"] == "MOVING3" and is_stopped():
                        add_log("Gear 2 Grabbed")
                        target_pose5 = [target_pose3[0], target_pose3[1], target_pose3[2]+0.09, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose5, 0.6, 0.6, True) # Move
                        shared_state["robot_moving"] = "MOVING4" 
                    elif shared_state["robot_moving"] == "MOVING4" and is_stopped():
                        add_log("Placing Gear 2")
                        place_pose = [0.671, -0.489, 0.212, 3.14159, 0.0, 0.0] # Offset added
                        rtde_control.moveL(place_pose, 0.6, 0.6, True) # Move
                        shared_state["robot_moving"] = "MOVING5" 
                    elif shared_state["robot_moving"] == "MOVING5" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED" 
                        gripper.open()
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Gear 2 Removed")
                        shared_state["robot_state"] = 12
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not pickup second gear: {e}")
        if shared_state["robot_state"] == 12: # pick up gear 3
                try:
                    if 'flip_magnitude' not in locals():
                        flip_magnitude = False
                    if shared_state["robot_moving"] == "IDLE" and is_stopped():   
                        add_log("Picking Up Gear 3")
                        gripper.open()
                        target_pose2 = [target_pose[0],target_pose[1],target_pose[2]+0.04,target_pose[3],target_pose[4],target_pose[5]] # above pose from earlier step
                        rtde_control.moveL(target_pose2, 0.25, 0.25, True) 
                        shared_state["robot_moving"] = "MOVING1"
                    elif shared_state["robot_moving"] == "MOVING1" and is_stopped():
                        shared_state["robot_moving"] = "MOVING2"
                        offset_magnitude = 0.03256
                        offset_magnitude_modified = 0  
                        if flip_magnitude:
                            offset_magnitude_modified = -offset_magnitude # go opposite direction at same angle
                        else:
                            offset_magnitude_modified = offset_magnitude # go original direction
                        current_pose = target_pose2 # do not get these from RTDE receive so that pausing will not mess up 
                        offset_x, offset_y = get_offsets(current_pose, offset_magnitude_modified,-0.84) # calculated experimentally 0.03217  -0.892
                        gripper.move(80, 10,100) # position, speed, force
                        target_pose3 = [current_pose[0]+offset_x, current_pose[1]+offset_y, current_pose[2], current_pose[3], current_pose[4], current_pose[5]]  # Offset added
                        rtde_control.moveL(target_pose3, 0.6, 0.6, True) # Move
                    elif shared_state["robot_moving"] == "MOVING2" and is_stopped():
                        shared_state["robot_moving"] = "MOVING3" 
                        current_pose = get_tcp_pose6(rtde_receive) # get current poses
                        target_pose4 = [target_pose3[0], target_pose3[1], target_pose3[2]-0.032, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose4, 0.6, 0.6, True) # Move
                        gripper.close()
                    elif shared_state["robot_moving"] == "MOVING3" and is_stopped():
                        time.sleep(1)
                        print(gripper._get_var("POS"))
                        if gripper._get_var("POS") > 220:
                            flip_magnitude = not flip_magnitude
                            shared_state["robot_moving"] = "IDLE"
                            add_log("Gear 3 Not Grabbed, Trying Again")
                        else:
                            add_log("Gear 3 Grabbed")
                            shared_state["robot_moving"] = "MOVING4" 
                    elif shared_state["robot_moving"] == "MOVING4" and is_stopped():
                        add_log("Gear 3 Lifted")
                        target_pose5 = [target_pose3[0], target_pose3[1], target_pose3[2]+0.09, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose5, 0.6, 0.6, True) # Move
                        shared_state["robot_moving"] = "MOVING5" 
                    elif shared_state["robot_moving"] == "MOVING5" and is_stopped():
                        add_log("Placing Gear 3")
                        place_pose = [0.631, -0.563, 0.212, 3.14159, 0.0, 0.0] # Offset added
                        rtde_control.moveL(place_pose, 0.2, 0.2, True) # Move
                        shared_state["robot_moving"] = "MOVING6" 
                    elif shared_state["robot_moving"] == "MOVING6" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED" 
                        gripper.open()
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Gear 3 Removed")
                        shared_state["robot_state"] = 13
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not pickup third gear: {e}")
        if shared_state["robot_state"] == 13: # pick up bottom half
                try:
                    if shared_state["robot_moving"] == "IDLE" and is_stopped():   
                        add_log("Picking Up Bottom Half")
                        gripper.open()
                        target_pose2 = [target_pose[0],target_pose[1],target_pose[2]+0.04,target_pose[3],target_pose[4],target_pose[5]] # above pose from earlier step
                        rtde_control.moveL(target_pose2, 0.25, 0.25, True) 
                        shared_state["robot_moving"] = "MOVING1"
                    elif shared_state["robot_moving"] == "MOVING1" and is_stopped():
                        shared_state["robot_moving"] = "MOVING2" 
                        current_pose = get_tcp_pose6(rtde_receive) # get current poses
                        target_pose4 = [target_pose3[0], target_pose3[1], target_pose3[2]-0.06, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose4, 0.6, 0.6, True) # Move
                        gripper.close()
                    elif shared_state["robot_moving"] == "MOVING2" and is_stopped():
                        add_log("Gear 3 Lifted")
                        target_pose5 = [target_pose3[0], target_pose3[1], target_pose3[2]+0.09, target_pose3[3], target_pose3[4], target_pose3[5]]  # Offset added
                        rtde_control.moveL(target_pose5, 0.6, 0.6, True) # Move
                        shared_state["robot_moving"] = "MOVING3" 
                    elif shared_state["robot_moving"] == "MOVING3" and is_stopped():
                        add_log("Placing Bottom Half")
                        place_pose = [0.522, -0.401, 0.182, 3.14159, 0.0, 0.0] # Offset added
                        rtde_control.moveL(place_pose, 0.2, 0.2, True) # Move
                        shared_state["robot_moving"] = "MOVING4" 
                    elif shared_state["robot_moving"] == "MOVING4" and is_stopped():
                        shared_state["robot_moving"] = "ARRIVED" 
                        gripper.open()
                    elif shared_state["robot_moving"] == "ARRIVED":
                        add_log("Bottom Half moved")
                        add_log("Sequence Finished! Resetting")
                        add_log("Press Start to Begin")
                        emergency_stop_event.set()
                        shared_state["robot_state"] = 1
                        shared_state["robot_moving"] = "IDLE"
                except Exception as e:
                    print(f"[WARN] Could not pickup bottom half: {e}")    
        time.sleep(0.1)


def main():
    try:
        # Connect to robot
        global rtde_receive
        rtde_receive = get_rtde_iface(UR_IP)
        global rtde_control
        rtde_control = get_rtde_control_iface(UR_IP)
    except Exception as e:
        print(f"[WARN] Could not connect to UR5e: {e}")
    global gripper
    gripper = RobotiqGripper()
    try:
        gripper.connect(hostname="192.168.1.5", port=63352)
        gripper.activate()
        print("[OK] Adaptive gripper connected and activated")
    except Exception as e:
        print(f"[WARN] Could not initialize gripper: {e}")
    try:
        # Load camera calibration
        if not T_TCP_CAM_PATH.exists():
            print(f"[ERROR] {T_TCP_CAM_PATH} not found. Run calibration first.")
            return
        
        # Load your model
        global model
        model = YOLO("06-09-2026.pt")

        global FRAMES_TO_CAPTURE
        FRAMES_TO_CAPTURE = 25  # Number of frames to capture for detection


        T_tcp_cam = np.load(T_TCP_CAM_PATH)
        print(f"[OK] Loaded {T_TCP_CAM_PATH}")
        
        # Setup RealSense
        ur5e_cam_pipeline = rs.pipeline()
        align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)
        config.enable_device("337122075128")  # UR5e RealSense serial 337122075128
        




        overview_cam_pipeline = rs.pipeline()
        overview_align = rs.align(rs.stream.color)
        overview_config = rs.config()
        overview_config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, FPS)
        overview_config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)
        overview_config.enable_device("327122076216")  # Overview RealSense serial


        realsense_available = True
    except Exception as e:
        print(f"[WARN] Could not initialize Realsense stream: {e}")

    try:
        
        shared_frame = {"ur5e": None, "overview": None,"captured_frame_buffer": [],
        "captured_frame_bbox": None,}
        shared_detections = {"objects": []}  # [(x, y, theta), ...]
        shared_state = {
        "robot_state": 1,
        "robot_moving": "IDLE",
        "fasteners_detected": 0,
        "gearbox_center": (None, None),
        "target_Xb": None,
        "target_Yb": None,
        "capturing_frames": False,
        "6D_pose": None,
        "source_pc": None,  
        "target_pc": None,  
        }

        frame_lock = threading.Lock()
        state_lock = threading.Lock()
        detection_lock = threading.Lock()
        emergency_stop_event = threading.Event() # for quick stop
        stop_event = threading.Event() # for ending program
        
        # --- ADD THIS BEFORE THREAD INITIALIZATION ---
        try:
            cfg = EngineConfig(
                resume_path="/home/robotics/Documents/human_robot_collab/gearbox/cnnlstm-Epoch-196-Loss-0.01737015192823795.pth", # UPDATE THIS
                annotation_path="/home/robotics/Documents/human_robot_collab/gearbox/intentpredictionattempt4-1/datasets/labels.json", # UPDATE THIS
                gpu=0 
            )

            # Callback to trigger the robot emergency stop
            def on_intent_flag(flag, conf):
                if flag == IntentEngine.FLAG_STOP:
                    add_log(f"Human Intent Stop! Conf: {conf:.2f}")
                    # emergency_stop_event.set()

            intent_engine = IntentEngine(cfg, on_flag_change=on_intent_flag)
            
            # WORKAROUND: Start ONLY the inference thread. 
            # We skip intent_engine.start() so CameraFactory doesn't crash or steal the RealSense port.
            intent_engine._running = True
            intent_engine._infer_thread = threading.Thread(
                target=intent_engine._inference_loop, daemon=True, name="IE-infer"
            )
            intent_engine._infer_thread.start()
            
        except Exception as e:
            print(f"[WARN] Intent Engine failed to initialize: {e}")
            intent_engine = None
        # ---------------------------------------------

        start_exception = []
        vision_thread = threading.Thread(
            target=realsense_loop,
            args=(shared_state, ur5e_cam_pipeline, align, shared_frame, shared_detections,
            frame_lock, detection_lock, stop_event, state_lock),
            daemon=True
        ) 
        # Update your thread initialization to pass the engine
        human_detection_thread = threading.Thread(
            target=human_detection_loop,
            args=(overview_cam_pipeline, overview_align, shared_frame,
            frame_lock, stop_event, emergency_stop_event, intent_engine),
            daemon=True
        )
        robot_movement_thread = threading.Thread(
            target=robot_loop,
            args=(shared_state, shared_detections, detection_lock, stop_event, emergency_stop_event, ur5e_cam_pipeline, shared_frame, state_lock, frame_lock),
            daemon=True
        ) 
        add_log("System Started")
        add_log("Threads Initialized")
        # Start both pipelines first
        ur5e_cam_pipeline.start(config)
        profile = ur5e_cam_pipeline.get_active_profile()
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        # 3. Adjust Exposure (Disable Auto-Exposure first!)
        # Auto-exposure is ON (1) by default. Set it to 0 (OFF) to use manual exposure.
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        # Set manual exposure time (in microseconds). 
        # E.g., 33000 is approximately 33 milliseconds. Increase this to brighten the scan.
        depth_sensor.set_option(rs.option.exposure, 33000)

        # 4. Adjust Laser Projector Intensity (Laser Power)
        # Range is typically 0 to 360mW. Nominal standard is 150mW. 
        # Crank this up to make the IR laser pattern more pronounced on dark surfaces.
        depth_sensor.set_option(rs.option.laser_power, 250.0)
        # overview_cam_pipeline.start(overview_config)
        time.sleep(2)

        # Then start worker threads
        vision_thread.start()           # UR5e camera processing
        # human_detection_thread.start()  # Overview camera processing
        robot_movement_thread.start()   # robot movement

        # make window clickable
        cv2.namedWindow(WINDOW_NAME)  
        params = {"emergency_stop": emergency_stop_event, "state_lock": state_lock, "shared_state":shared_state}
        cv2.setMouseCallback(WINDOW_NAME, button_callback, params)
       

        while True:
            
            if realsense_available:
                with frame_lock:
                    r_frame1 = shared_frame["ur5e"].copy() if shared_frame["ur5e"] is not None else None
                    if r_frame1 is not None:
                        r_frame1 = cv2.resize(r_frame1, (HALF_WIDTH, HALF_HEIGHT))
                    r_frame2 = shared_frame["overview"].copy() if shared_frame["overview"] is not None else None
                    if r_frame2 is not None:
                        r_frame2 = cv2.resize(r_frame2, (HALF_WIDTH, HALF_HEIGHT))
            else:
                r_frame1 = None
                r_frame2 = None
            


            ui_frame = create_ui_frame(r_frame1, r_frame2)
            cv2.imshow(WINDOW_NAME, ui_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise Exception("q key was hit")
            # Listen for 's' key on the main window thread
            elif key == ord('s') or key == ord('S'):
                with state_lock:
                    # Only trigger if we are in State 3 and not already capturing
                    if shared_state.get("robot_state") == 3 and not shared_state.get("capturing_frames", False):
                        print("\n[+] 's' key pressed! Starting burst sequence...")
                        shared_state["capturing_frames"] = True
                        with frame_lock:
                            shared_frame["captured_frame_buffer"] = [] # Clear out old buffers
            with state_lock:
                if shared_state.get("6D_pose") is not None and shared_state.get("robot_state") == 3:
                    computed_pose = shared_state["6D_pose"]
                    print(f"[INFO] 6D Pose: {computed_pose}")
                    if shared_state["source_pc"] is None:
                        add_log("SOURCE is None, cannot draw registration result.")
                    if shared_state["target_pc"] is None:
                        add_log("TARGET is None, cannot draw registration result.")
                    if shared_state["source_pc"] and shared_state["target_pc"] is not None:
                        add_log("Drawing registration result...")
                        gearbox_3D_Full_Pipeline.draw_registration_result(shared_state["source_pc"], shared_state["target_pc"], computed_pose)

    except Exception as e:
        try:
            pass
        except NameError:
            print("Variable does not exist.")
        print(f"[WARN] Could not run: {e}")
        stop_event.set()
        try:    
            vision_thread.join()
        except Exception as e:
            print("Are both realsense cameras plugged in?")
        cv2.destroyAllWindows()
        traceback.print_exc()

main()
