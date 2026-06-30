import pyrealsense2 as rs

# Initialize the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams (adjust resolutions as needed)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline and capture the active profile
profile = pipeline.start(config)

try:
    # 1. Fetch the stream profiles
    depth_stream = profile.get_stream(rs.stream.depth)
    color_stream = profile.get_stream(rs.stream.color)

    # 2. Downcast to video stream profiles
    depth_video_profile = depth_stream.as_video_stream_profile()
    color_video_profile = color_stream.as_video_stream_profile()

    # 3. Retrieve intrinsic attributes
    depth_intrinsics = depth_video_profile.get_intrinsics()
    color_intrinsics = color_video_profile.get_intrinsics()

    # Print Depth Intrinsics
    print("--- Depth Intrinsics ---")
    print(f"Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
    print(f"Focal Length (fx, fy): ({depth_intrinsics.fx}, {depth_intrinsics.fy})")
    print(f"Principal Point (cx, cy): ({depth_intrinsics.ppx}, {depth_intrinsics.ppy})")
    print(f"Distortion Model: {depth_intrinsics.model}")
    print(f"Distortion Coefficients: {depth_intrinsics.coeffs}")

    # Print Color Intrinsics
    print("\n--- Color Intrinsics ---")
    print(f"Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
    print(f"Focal Length (fx, fy): ({color_intrinsics.fx}, {color_intrinsics.fy})")
    print(f"Principal Point (cx, cy): ({color_intrinsics.ppx}, {color_intrinsics.ppy})")
    print(f"Distortion Model: {color_intrinsics.model}")
    print(f"Distortion Coefficients: {color_intrinsics.coeffs}")

finally:
    # Stop streaming
    pipeline.stop()
