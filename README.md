# YOLOE-Voxeland

ROS2 package for YOLOE object detection and segmentation integrated with Voxeland.

## Setup

### 1. Create Virtual Environment and Install Dependencies

```bash
cd ~/ros2_ws/src/YOLOE-Voxeland
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Important:** Always activate the venv before running YOLOE:
```bash
source ~/ros2_ws/src/YOLOE-Voxeland/venv/bin/activate
```

### 2. Download YOLOE Models

```bash
cd ~/ros2_ws/src/YOLOE-Voxeland/src/yoloe_ros2/yoloe_ros2
python download_yoloe.py

# Download additional files manually:
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
wget https://raw.githubusercontent.com/THU-MIG/yoloe/main/tools/ram_tag_list.txt
```

## Usage (Without ROS2)

**Run inference on images:**
```bash
cd ~/ros2_ws/src/YOLOE-Voxeland/src/yoloe_ros2/yoloe_ros2
source ~/ros2_ws/src/YOLOE-Voxeland/venv/bin/activate
python yoloe_main.py -img image1.jpg image2.png
```

`yoloe_main.py` uses `yoloe_inference.py` to run inference. All input images must be stored in the `src/yoloe_ros2/yoloe_ros2/input_images` directory.


## Usage with [Voxeland](https://github.com/MAPIRlab/Voxeland)

### 1. Build the Workspace

Clean previous build artifacts and execute `colcon build` to compile the workspace. Please adapt the paths if necessary:

```bash
cd ~/ros2_ws
rm -rf build/ install/ log/
colcon build --symlink-install --cmake-clean-cache
```

### 2. Launch the Detectron2 ROS 2 Node

Please adapt the paths if necessary:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run yoloe_ros2 yoloe_node
```

### 3. Run Voxeland and Play a ScanNet ROS Bag

Create and execute a bash script that contains the following commands. Please adapt the paths if necessary:

```bash
cd ~/ros2_ws

# Init voxeland_robot_perception with YOLOE detector
gnome-terminal -- bash -c "source ~/.bashrc; source /home/ubuntu/ros2_ws/venvs/voxenv/bin/activate; ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=yoloe; exec bash"

# Init voxeland server
gnome-terminal -- bash -c "ros2 launch voxeland voxeland_server.launch.xml; exec bash"

# Open bag folder and play ros2 bag
gnome-terminal -- bash -c "cd /home/ubuntu/ros2_ws/bag/ScanNet/to_ros/ROS2_bags/scene0000_01/; ros2 bag play scene0000_01.db3; exec bash"
```

### Service Interface

**Service:** `/yoloe/segment`  
**Type:** `segmentation_msgs/srv/SegmentImage`

**Request:**
- `sensor_msgs/Image image` - Input RGB image

**Response:**
- `segmentation_msgs/SemanticInstance2D[] instances` - Detected objects with masks and bounding boxes

**Visualization Topic:** `/segmentedImage` (if enabled)

## Configuration

Edit `src/yoloe_ros2/yoloe_ros2/config.json`:
```json
{
    "save_output_files": false
}
```

- `save_output_files: true` - Save detection results, masks, and visualizations to disk
- `save_output_files: false` - Return results in memory only (recommended for ROS2)
