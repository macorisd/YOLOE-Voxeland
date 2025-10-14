import warnings
warnings.filterwarnings('ignore', message='The value of the smallest subnormal')

import numpy as np
import cv2
import time
from typing import Tuple, Dict, List

from yoloe_main import YOLOE


class YOLOERos2(YOLOE):
    """
    YOLOE pipeline for ROS2.
    """
    def __init__(self):
        # Initialize parent class without command line arguments
        # Use default input_images_dir
        super().__init__(input_images_dir='input_images')
    
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
