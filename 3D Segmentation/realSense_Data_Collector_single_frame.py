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

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Enable High-Resolution streams (1280x720 is standard for D400 series)
    # If the camera rejects this, change to 640x480
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    print("[*] Starting pipeline...")
    profile = pipeline.start(config)

    # 3. Create an align object (CRITICAL for your 2D-to-3D crop architecture)
    # This maps the depth pixels to the exact same physical space as the color pixels
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Give the camera auto-exposure a second to settle
    time.sleep(1)

    frame_count = 0
    print("[*] Camera ready. Press 's' to save a frame, 'q' to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Render the image to the screen to see what you are doing
            cv2.imshow("RealSense Capture (Press 's' to Save, 'q' to Quit)", color_image)
            
            # Key listener
            key = cv2.waitKey(1)

            # If 's' is pressed, save the data
            if key == ord('s'):
                # Format filenames (e.g., frame_0001_rgb.png)
                base_name = os.path.join(save_dir, f"frame_{frame_count:04d}")
                
                rgb_filename = f"{base_name}_rgb.png"
                depth_npy_filename = f"{base_name}_depth.npy"
                depth_png_filename = f"{base_name}_depth.png"

                # Save RGB as 8-bit PNG
                cv2.imwrite(rgb_filename, color_image)
                
                # Save Depth as Raw Numpy Array (Highly recommended for Open3D ICP)
                np.save(depth_npy_filename, depth_image)
                
                # Save Depth as 16-bit PNG (Optional, but good for visual debugging)
                cv2.imwrite(depth_png_filename, depth_image)

                print(f"[+] Saved {rgb_filename} and matched depth files.")
                frame_count += 1

                # Visual flash to confirm capture
                flash = np.ones_like(color_image) * 255
                cv2.imshow("RealSense Capture (Press 's' to Save, 'q' to Quit)", flash)
                cv2.waitKey(50)

            # If 'q' or 'ESC' is pressed, close the program
            elif key == ord('q') or key == 27:
                print("[*] Quitting and safely shutting down camera...")
                break

    finally:
        # Stop streaming safely
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()