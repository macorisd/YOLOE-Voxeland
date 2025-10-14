from typing import List, Tuple, Union, Optional
import time
import argparse
import os
import cv2
from yoloe_inference import YOLOEInference


class YOLOE:
    def __init__(self, input_images_dir: str = 'input_images'):
        """
        Initialize YOLOE pipeline.
        
        Args:
            input_images_dir: Directory where input images are located
        """
        self.input_images_dir = input_images_dir
        self.yoloe_inference = YOLOEInference()

    def run(self, input_image_names: Union[str, List[str]]) -> Tuple[float, Optional[float]]:
        """
        Run inference on one or more images.
        
        Args:
            input_image_names: Single image name or list of image names
            
        Returns:
            Tuple of (total_time, average_time)
        """
        if isinstance(input_image_names, str):
            input_image_names = [input_image_names]

        total_time = 0
        total_runs = len(input_image_names)
        
        for i, image_name in enumerate(input_image_names):
            start_time = time.time()
            print(f"\n[YOLOE] Running inference for image {i+1}/{total_runs}: {image_name}...")

            # Build full path to image
            image_path = os.path.join(self.input_images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"[YOLOE] Error: Image not found at {image_path}")
                continue
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[YOLOE] Error: Failed to load image {image_path}")
                continue
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference with image name for unique output folder
            results = self.yoloe_inference.run(image_rgb, image_name=image_name)
            
            print(f"\n[YOLOE] Detected {results['num_detections']} instances")
            print(f"\n[YOLOE] Categories found: {', '.join(results['categories'])}")
            print(f"\n[YOLOE] Results saved to: {results['output_dir']}")

            # Calculate execution time
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"\n[YOLOE] Finished in {elapsed:.2f} seconds.")

        if total_runs > 1:
            average_time = total_time / total_runs
            print(f"\n[YOLOE] Total execution time for {total_runs} executions: {total_time:.2f} seconds.")
            print(f"\n[YOLOE] Average execution time: {average_time:.2f} seconds.")
        else:
            average_time = None

        print("\n[YOLOE] All images processed successfully.")
        return total_time, average_time


def main(input_image_names: Union[str, List[str]]):
    if not input_image_names:
        raise ValueError("\n[YOLOE] No input image names provided. Please provide a list of image names.")

    yoloe = YOLOE()
    yoloe.run(input_image_names)


if __name__ == "__main__":
    # ArgumentParser to handle command line arguments
    parser = argparse.ArgumentParser(description="Run the YOLOE pipeline on specified images.")
    
    # Add an argument for input images
    parser.add_argument(
        "-img",
        "--input_images",
        nargs='*', # Zero or more arguments
        default=['desk.jpg'], # Default image if none are provided
        help="One or more input image names (e.g., image1.png image2.jpg). Defaults to ['desk.jpg'] if not specified. Images must be located in the 'input_images' directory.",
        metavar="IMAGE_NAME"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(input_image_names=args.input_images)
