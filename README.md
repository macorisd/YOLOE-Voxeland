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

## Usage

### Standalone (Without ROS2)

**Run inference on images:**
```bash
cd ~/ros2_ws/src/YOLOE-Voxeland/src/yoloe_ros2/yoloe_ros2
source ~/ros2_ws/src/YOLOE-Voxeland/venv/bin/activate
python yoloe_main.py -img image1.jpg image2.png
```

`yoloe_main.py` uses `yoloe_inference.py` to run inference. All input images must be stored in the `src/yoloe_ros2/yoloe_ros2/input_images` directory.


### With ROS2

**Build the package:**
```bash
cd ~/ros2_ws
colcon build --packages-select yoloe_ros2 --symlink-install --cmake-clean-cache
```

**Run the YOLOE node:**
```bash
cd ~/ros2_ws/src/YOLOE-Voxeland/src/yoloe_ros2/yoloe_ros2
source ~/ros2_ws/src/YOLOE-Voxeland/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/YOLOE-Voxeland/venv/lib/python3.10/site-packages
source ~/ros2_ws/install/setup.bash
ros2 run yoloe_ros2 yoloe_node
```

## Integration with Voxeland (for 3D semantic maps)

### Launch Voxeland with YOLOE

**Terminal 1 - Voxeland Robot Perception with YOLOE:**
```bash
cd ~/ros2_ws
source ~/.bashrc
source ~/ros2_ws/venvs/voxenv/bin/activate
ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=yoloe
```

**Terminal 2 - Voxeland Server:**
```bash
ros2 launch voxeland voxeland_server.launch.xml
```

**Terminal 3 - Play ROS2 Bag:**
```bash
cd ~/ros2_ws/bag/ScanNet/to_ros/ROS2_bags/scene0000_01/
ros2 bag play scene0000_01.db3
```

### Service Interface

**Service:** `/yoloe/segment`  
**Type:** `segmentation_msgs/srv/SegmentImage`

**Request:**
- `sensor_msgs/Image image` - Input RGB image

**Response:**
- `segmentation_msgs/SemanticInstance2D[] instances` - Detected objects with masks and bounding boxes

**Visualization Topic:** `/yoloe/segmentedImage` (if enabled)

## Configuration

Edit `src/yoloe_ros2/yoloe_ros2/config.json`:
```json
{
    "save_output_files": false
}
```

- `save_output_files: true` - Save detection results, masks, and visualizations to disk
- `save_output_files: false` - Return results in memory only (recommended for ROS2)
