import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import time

# ==============================================================================
# CONFIGURATION & CAMERA INTRINSICS (From your Gearbox Pipeline)
# ==============================================================================
# RealSense D435 Default Intrinsics
INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
    width=640, 
    height=480, 
    fx=609.3075561523438,  
    fy=608.9049072265625,  
    cx=326.06304931640625, 
    cy=249.21212768554688  
)

def main():
    # 1. Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("[*] Starting RealSense pipeline...")
    pipeline.start(config)

    # Create an align object mapping depth to color space
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Let camera auto-exposure settle
    time.sleep(1)

    # State tracking variables for frame accumulation
    gathering_data = False
    depth_burst_buffer = []
    captured_color_image = None

    print("[*] Camera ready. Press 's' to capture burst, 'q' to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Sever connection to SDK internal C++ pools
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()

            # Sequential burst accumulation state machine
            if gathering_data:
                depth_burst_buffer.append(depth_image)
                
                # Capture the 10th frame's RGB data to anchor texture alignment
                if len(depth_burst_buffer) == 10:
                    captured_color_image = color_image.copy()
                
                # Overlay real-time progress text onto preview
                cv2.putText(color_image, f"GATHERING BURST: {len(depth_burst_buffer)}/20", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Once we hit 20 frames, drop out of live stream to process and visualize
                if len(depth_burst_buffer) == 20:
                    print("\n[+] Burst captured! Processing data...")
                    break

            # Render live tracking view
            cv2.imshow("RealSense Live View (Press 's' to Capture, 'q' to Quit)", color_image)
            key = cv2.waitKey(1)

            if key == ord('s') or key == ord('S'):
                if not gathering_data:
                    print("[*] Activating 20-frame natural burst grab...")
                    gathering_data = True

            elif key == ord('q') or key == 27:
                print("[*] User aborted pipeline.")
                return

    finally:
        # Shut down camera stream cleanly immediately after data loop breaks
        print("[*] Releasing RealSense hardware pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

    # --------------------------------------------------------------------------
    # PROCESSING & VISUALIZATION STAGE
    # --------------------------------------------------------------------------
    
    # 1. Compute Temporal Median Filter
    print("[*] Computing Temporal Median across burst stack...")
    depth_stack = np.stack(depth_burst_buffer, axis=0)
    median_depth_image = np.median(depth_stack, axis=0).astype(np.uint16)

    # Fallback guard for the color image
    if captured_color_image is None:
        captured_color_image = color_image

    # 2. Display the 2D PNG Images (RGB and Colorized Depth)
    print("[*] Displaying 2D captured maps. Press ANY KEY on the image window to proceed to 3D point cloud.")
    
    # Generate a readable depth visual map (Jet colormap)
    colorized_depth = cv2.applyColorMap(
        cv2.convertScaleAbs(median_depth_image, alpha=0.03), 
        cv2.COLORMAP_JET
    )
    
    # Present both 2D dimensions
    cv2.imshow("Captured PNG - RGB", captured_color_image)
    cv2.imshow("Captured PNG - Median Depth Map", colorized_depth)
    cv2.waitKey(0)  # Wait indefinitely until user acknowledges
    cv2.destroyAllWindows()

    # 3. Build RGBD Image and Project 3D Point Cloud (Gearbox Pipeline Style)
    print("[*] Projecting maps into a textured 3D Point Cloud...")
    
    # Open3D expects standard RGB instead of OpenCV's default BGR
    color_rgb = cv2.cvtColor(captured_color_image, cv2.COLOR_BGR2RGB)
    
    o3d_color = o3d.geometry.Image(color_rgb)
    o3d_depth = o3d.geometry.Image(median_depth_image)
    
    # Map the color map cleanly over the depth frame array
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, 
        o3d_depth, 
        depth_scale=1000.0, 
        depth_trunc=2.0, 
        convert_rgb_to_intensity=False
    )
    
    # Generate point cloud using your exact D435 camera intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, INTRINSICS)
    
    print(f"[+] Generated point cloud containing {len(pcd.points)} points.")
    print("[*] Opening interactive 3D view. Close the Open3D window to exit script.")
    
    # Display the final textured point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud Projection")
    
    print("[*] Pipeline closed successfully.")

if __name__ == "__main__":
    main()