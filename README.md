# hrc-robotic-disassembly
A modular ROS 2 framework for automated robotic disassembly, featuring containerized deployment, custom 3D asset integration, and specialized Python-based perception and manipulation nodes.

## Quickstart & Execution

Follow these steps from the root directory (`gearbox-perception-system`) to build, run, and visualize the system.

### 1. Navigate to the Docker Directory
From the root of the project repository, move into the Docker configuration folder:
```bash
cd ROS2/docker
```

### 2. Build and Launch the Docker Container
Start the container environment using Docker Compose:
```bash
docker compose up --build -d
```

### 3. Enter the Docker Shell
Open an interactive bash shell inside the running container:
```bash
docker exec -it ros2_dev_container /bin/bash
```

### 4. Build the ROS2 Workspace
Inside the container shell (navigate to the workspace root if needed, e.g., /ros2_ws):
```bash
colcon build
```

### 5. Source the Environment
Overlay the newly built packages onto your current active ROS2 workspace:
```bash
source install/setup.bash
```

### 6. Run the Launch File
Execute the primary perception and control system launch file:
```bash
ros2 launch gearbox_perception perception_pipeline.launch.py
```

### 7. Connect to Foxglove Studio
1. Open Foxglove Studio on your host machine either through the website or application.

2. Select Open Connection.

3. Choose Foxglove WebSocket.

4. Set the WebSocket URL using your configured port:
```
ws://localhost:8765
```
5. Click Connect to view live point clouds, camera streams, and ROS2 topic telemetry.
