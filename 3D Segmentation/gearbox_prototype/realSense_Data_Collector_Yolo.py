import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO  # Run: pip install ultralytics

def main():
    # 1. Setup the save directory
    save_dir = "realsense_lab_data"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[*] Saving data to directory: ./{save_dir}")

    # 2. Load the YOLO Model 
    # NOTE: Replace "yolov8n.pt" with your custom gearbox weights path (e.g., "best.pt")
    model_path = "06-09-2026.pt"
    print(f"[*] Loading YOLO model from {model_path}...")
    yolo_model = YOLO(model_path)

    # 3. Configure RealSense depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    print("[*] Starting RealSense pipeline...")
    profile = pipeline.start(config)

    # Create an align object mapping depth to color space
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Give the camera auto-exposure a second to settle
    time.sleep(1)

    frame_count = 0
    
    # State tracking variables for sequential frame accumulation
    gathering_data = False
    depth_burst_buffer = []
    captured_color_image = None
    saved_crop_box = None  # Holds locked (x1, y1, x2, y2) coordinates

    print("[*] Camera ready. Press 's' to save a frame, 'q' to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Force memory copies to completely sever connections to the C++ hardware pool
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()

            # ------------------------------------------------------------------
            # LIVE INFERENCE: Run YOLO on the live RGB stream
            # ------------------------------------------------------------------
            yolo_results = yolo_model(color_image, conf=0.60, verbose=False)[0]
            # Create a preview frame that draws the live bounding boxes and labels
            preview_image = yolo_results.plot()

            # ------------------------------------------------------------------
            # SEQUENTIAL DATA ACCUMULATION STATE MACHINE
            # ------------------------------------------------------------------
            if gathering_data:
                depth_burst_buffer.append(depth_image)
                
                # Capture the 10th frame's RGB data to anchor texture alignment
                if len(depth_burst_buffer) == 10:
                    captured_color_image = color_image.copy()
                
                # Draw a real-time progress text overlay onto the live preview screen
                cv2.putText(preview_image, f"GATHERING BURST: {len(depth_burst_buffer)}/20", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Once we successfully have 20 frames, compute median, crop, and save
                if len(depth_burst_buffer) == 20:
                    print(f"[*] Processing 20 accumulated frames for Set {frame_count:04d}...")
                    
                    # Stack frames: Shape (20, 480, 640)
                    depth_stack = np.stack(depth_burst_buffer, axis=0)
                    
                    # Compute temporal median and cast back to native uint16 depth
                    median_depth_image = np.median(depth_stack, axis=0).astype(np.uint16)

                    # Fallback guard for the color image
                    if captured_color_image is None:
                        captured_color_image = color_image

                    # ----------------------------------------------------------
                    # BOUNDING BOX CROP EXECUTION
                    # ----------------------------------------------------------
                    x1, y1, x2, y2 = saved_crop_box
                    
                    # Slice out only the detected bounding box region
                    cropped_color = captured_color_image[y1 - 30:y2 + 30, x1 - 30:x2 + 30]
                    cropped_depth = median_depth_image[y1 - 30:y2 + 30, x1 - 30:x2 + 30]

                    # Format filenames
                    base_name = os.path.join(save_dir, f"frame_{frame_count:04d}")
                    rgb_filename = f"{base_name}_cropped_rgb.png"
                    depth_npy_filename = f"{base_name}_cropped_depth.npy"
                    depth_png_filename = f"{base_name}_cropped_depth.png"

                    # Write out ONLY the cropped objects to disk
                    cv2.imwrite(rgb_filename, cropped_color)
                    np.save(depth_npy_filename, cropped_depth)
                    cv2.imwrite(depth_png_filename, cropped_depth)

                    print(f"[+] Successfully saved cropped matrices (Size: {cropped_depth.shape}) to {depth_npy_filename}!")
                    frame_count += 1

                    # Reset state machine variables for next capture trigger
                    gathering_data = False
                    depth_burst_buffer = []
                    captured_color_image = None
                    saved_crop_box = None

                    # Flash visual confirmation frame
                    flash = np.ones_like(preview_image) * 255
                    cv2.imshow("RealSense YOLO Capture (Press 's' to Save, 'q' to Quit)", flash)
                    cv2.waitKey(75)
                    continue

            # Render the preview image containing live YOLO bounding boxes to screen
            cv2.imshow("RealSense YOLO Capture (Press 's' to Save, 'q' to Quit)", preview_image)
            
            # Key listener
            key = cv2.waitKey(1)

            # Trigger frame accumulation on 's' press
            if key == ord('s') and not gathering_data:
                # CRITICAL: Verify YOLO actually sees an object before starting capture
                if len(yolo_results.boxes) > 0:
                    # Select the highest confidence object detection bounding box
                    best_box = yolo_results.boxes[0]
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    
                    # Store the box coordinates to apply to the final median map later
                    saved_crop_box = (x1, y1, x2, y2)
                    
                    print(f"[*] Locked YOLO Target Box: Bounding Coordinates [{x1}, {y1}, {x2}, {y2}]")
                    print(f"[*] Activating natural burst frame grab for Set {frame_count:04d}...")
                    gathering_data = True
                else:
                    print("[-] CAPTURE REJECTED: YOLO does not detect the gearbox object in this frame view!")

            # Safe exit
            elif key == ord('q') or key == 27:
                print("[*] Quitting and safely shutting down camera...")
                break

    finally:
        # Stop streaming safely and release native allocations
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()