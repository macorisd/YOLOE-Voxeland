import warnings
warnings.filterwarnings('ignore', message='The value of the smallest subnormal')

import numpy as np
import cv2
import time
import json
import os
from typing import Tuple, Dict

from yoloe_ros2.yoloe_main import YOLOE


class YOLOERos2(YOLOE):
    """
    YOLOE class for ROS2.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize YOLOE for ROS2.
        
        Args:
            config_path: Path to config.json file. If None, uses default path.
        """
        # Load configuration
        if config_path is None:
            # Default path is in the same directory as this file
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        self.config = self._load_config(config_path)
        self.save_output_files = self.config.get('save_output_files', False)
        
        print(f"[YOLOERos2] Configuration loaded: save_output_files={self.save_output_files}")
        
        super().__init__(input_images_dir='input_images', save_output_files=self.save_output_files)
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config.json file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            print(f"[YOLOERos2] Warning: Config file not found at {config_path}, using defaults")
            return {'save_output_files': False}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"[YOLOERos2] Configuration loaded from {config_path}")
            return config
        except Exception as e:
            print(f"[YOLOERos2] Error loading config file: {e}, using defaults")
            return {'save_output_files': False}
    
    def run(self, input_image: np.ndarray) -> Tuple[Dict, float]:
        """
        Run YOLOE inference on an input image (for ROS2).
        
        Args:
            input_image: Input image as numpy array (BGR format from ROS/OpenCV)
            
        Returns:
            Tuple of (results_dict, total_time)
        """
        start_time = time.time()

        # Convert BGR to RGB (ROS uses BGR, YOLOE expects RGB)
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Run inference without saving to timestamped folders (ROS mode)
        results = self.yoloe_inference.run(image_rgb, image_name=None)

        # Calculate execution time
        total_time = time.time() - start_time

        return results, total_time


def main():
    """Test function with a dummy image."""
    input_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy image for testing
    pipeline = YOLOERos2()

    results, total_time = pipeline.run(input_image)
    print(f"Results: {results}")
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
