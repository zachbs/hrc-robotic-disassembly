# Start from official Ubuntu 22.04 base as planned for L4T 36.5.0 / JetPack 6.2.2
# This is the base image
FROM ubuntu:22.04 
# Set non-interactive mode for apt-get to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive 

# Install essential build tools alongside X11 GUI rendering engine frameworks
# "apt-get update" updates the package lists for upgrades and new package installations, 
# while "apt-get install -y" installs the specified packages without prompting for confirmation.
# rm -rf /var/lib/apt/lists/* removes the list of all package files downloaded by apt-get update
# to free up space in the Docker image.
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    cmake \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libegl1 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble Base System Components using modern signed-by keyrings
RUN mkdir -p /usr/share/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip inside global context safely
RUN pip3 install --no-cache-dir --upgrade pip

# Install exact python pinned library components 
# NOTE: If executing directly on ARM64/Jetson, you may need to swap open3d and pyrealsense2 
# to build-from-source layers or use alternative community wheel paths.
RUN pip3 install --no-cache-dir \
    ur_rtde==1.6.2 \
    scipy==1.13.1 \
    numpy==1.26.1 \
    PyYAML==6.0.2 \
    ultralytics==8.3.169 \
    open3d-unofficial-arm==0.19.0.post9 \
    pyrealsense2-beta==2.57.6.10172

# Set up reliable runtime sourcing via an entrypoint script
# This script ensures that the ROS 2 environment is properly sourced before executing any command passed to the container. This 
# means that when you run the container, it will automatically set up the ROS 2 environment, allowing you to use ROS 2 commands 
# and tools without needing to manually source the setup script each time.
RUN echo '#!/bin/bash\nset -e\nsource /opt/ros/humble/setup.bash\nexec "$@"' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

#this is where the workspace will be mounted in the container, and where the user will be dropped into when the container is run
WORKDIR /workspace

# This makes it so the Path and environment variables are set up correctly for ROS 2, allowing you to run ROS 2 commands and tools 
# without needing to manually source the setup script each time.
ENTRYPOINT ["/ros_entrypoint.sh"]
# The CMD instruction specifies the default command to run when the container starts. In this case, it starts a bash shell, allowing you to interact with the container's environment.
CMD ["bash"]