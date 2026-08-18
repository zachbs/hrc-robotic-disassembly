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
If the container has already been built previously just use this command to start the container:
```bash
docker compose up -d
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

### 8. Exiting the Program
1. To exit out of ROS2 first press the CTRL key and C key at the same time
2. Next type this in the terminal
```bash
exit
```
3. Lastly to shutdown docker run this:
```bash
docker compose down
```

### If you are on a Windows device then Docker and WSL could still be running in the background
If you see both are still running, go to the upward arrow (show hidden icons button) on the task bar
and find the Docker app icon. Right click and hit "Quit Docker Desktop".
This should stop WSL and Docker from running in the background but there is a chance WSL is still running.
To see if WSL is still running enter this:
```bash
wsl --list --running
```
If you see something like:
docker-desktop
docker-desktop-data
Ubuntu

or
Ubuntu

then enter this command:
```bash
wsl --shutdown
```
This will shut down WSL and free up a lot of memory being used.
