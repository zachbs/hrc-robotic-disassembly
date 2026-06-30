import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def main():
    # 1. Setup the save directory
    save_dir = "realsense_lab_data"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[*] Saving data to directory: ./{save_dir}")

    # 2. Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    print("[*] Starting pipeline...")
    profile = pipeline.start(config)

    # 3. Create an align object mapping depth to color space
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Give the camera auto-exposure a second to settle
    time.sleep(1)

    frame_count = 0
    
    # State tracking variables for sequential frame accumulation
    gathering_data = False
    depth_burst_buffer = []
    captured_color_image = None

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

            # ------------------------------------------------------------------
            # CRITICAL FIX: Add .copy() to completely sever the connection
            # to the RealSense SDK's internal C++ hardware frame pool.
            # ------------------------------------------------------------------
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()

            # ------------------------------------------------------------------
            # SEQUENTIAL DATA ACCUMULATION STATE MACHINE
            # ------------------------------------------------------------------
            if gathering_data:
                depth_burst_buffer.append(depth_image)
                
                # Capture the 10th frame's RGB data to anchor texture alignment
                if len(depth_burst_buffer) == 10:
                    captured_color_image = color_image.copy()
                
                # Draw a real-time progress text overlay onto the preview frame
                cv2.putText(color_image, f"GATHERING BURST: {len(depth_burst_buffer)}/20", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Once we successfully have 20 frames, process and save them
                if len(depth_burst_buffer) == 20:
                    print(f"[*] Processing 20 accumulated frames for Set {frame_count:04d}...")
                    
                    # Stack frames: Shape (20, 480, 640)
                    depth_stack = np.stack(depth_burst_buffer, axis=0)
                    
                    # Compute temporal median and cast back to native uint16 depth
                    median_depth_image = np.median(depth_stack, axis=0).astype(np.uint16)

                    # Format filenames
                    base_name = os.path.join(save_dir, f"frame_{frame_count:04d}")
                    rgb_filename = f"{base_name}_rgb.png"
                    depth_npy_filename = f"{base_name}_depth.npy"
                    depth_png_filename = f"{base_name}_depth.png"

                    # Fallback guard for the color image
                    if captured_color_image is None:
                        captured_color_image = color_image

                    # Write out filtered maps to disk
                    cv2.imwrite(rgb_filename, captured_color_image)
                    np.save(depth_npy_filename, median_depth_image)
                    cv2.imwrite(depth_png_filename, median_depth_image)

                    print(f"[+] Successfully saved median-filtered array to {depth_npy_filename}!")
                    frame_count += 1

                    # Reset state machine variables for next capture trigger
                    gathering_data = False
                    depth_burst_buffer = []
                    captured_color_image = None

                    # Flash visual confirmation frame
                    flash = np.ones_like(color_image) * 255
                    cv2.imshow("RealSense Capture (Press 's' to Save, 'q' to Quit)", flash)
                    cv2.waitKey(75)
                    continue

            # Render the current frame image to the screen
            cv2.imshow("RealSense Capture (Press 's' to Save, 'q' to Quit)", color_image)
            
            # Key listener
            key = cv2.waitKey(1)

            # Trigger frame accumulation on 's' press
            if key == ord('s') and not gathering_data:
                print(f"[*] Activating natural burst frame grab for Set {frame_count:04d}...")
                gathering_data = True

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