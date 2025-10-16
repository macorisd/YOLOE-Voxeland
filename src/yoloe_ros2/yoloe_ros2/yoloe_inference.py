from ultralytics import YOLOE
import json
import os
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Tuple


class YOLOEInference:
    """
    Class for performing inference with YOLOE model and saving organized outputs.
    """
    
    def __init__(self, model_yaml: str = "yoloe-v8l.yaml", 
                 model_weights: str = "yoloe-v8l-seg.pt",
                 fused_weights: str = "yoloe-v8l-seg-pf.pt",
                 vocab_file: str = "ram_tag_list.txt",
                 output_base_dir: str = "output",
                 save_output_files: bool = True):
        """
        Initialize YOLOE inference model.
        
        Args:
            model_yaml: Path to model configuration YAML
            model_weights: Path to model weights
            fused_weights: Path to fused model weights
            vocab_file: Path to vocabulary file with class names
            output_base_dir: Base directory for outputs
            save_output_files: Whether to save output files to disk
        """
        self.output_base_dir = output_base_dir
        self.save_output_files = save_output_files
        
        # Load unfused model to get vocabulary
        print("Loading unfused model...")
        unfused_model = YOLOE(model_yaml).cuda()
        unfused_model.load(model_weights)
        unfused_model.eval()
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, 'r') as f:
            self.names = [x.strip() for x in f.readlines()]
        vocab = unfused_model.get_vocab(self.names)
        
        # Load fused model
        print("Loading fused model...")
        self.model = YOLOE(fused_weights).cuda()
        self.model.set_vocab(vocab, names=self.names)
        self.model.model.model[-1].is_fused = True
        self.model.model.model[-1].conf = 0.001
        self.model.model.model[-1].max_det = 1000
        
        print("YOLOE model initialized successfully!")
    
    def _scale_mask_to_original(self, mask_640: np.ndarray, orig_shape: Tuple[int, int], 
                                model_input_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Scale a mask from model input size to original image size, accounting for letterbox padding.
        
        Args:
            mask_640: Mask in model input space (640x640)
            orig_shape: Original image shape (height, width)
            model_input_size: Model input size (height, width), default (640, 640)
            
        Returns:
            Mask scaled to original image size
        """
        orig_h, orig_w = orig_shape
        model_h, model_w = model_input_size
        
        # Calculate the scale factor used in letterbox
        scale = min(model_h / orig_h, model_w / orig_w)
        
        # Calculate the new shape after scaling (before padding)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Calculate padding
        pad_h = (model_h - new_h) // 2
        pad_w = (model_w - new_w) // 2
        
        # Remove padding from the mask
        mask_unpadded = mask_640[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        
        # Resize back to original dimensions
        mask_original = cv2.resize(
            mask_unpadded.astype(np.float32),
            (orig_w, orig_h),  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        
        return mask_original
    
    def run(self, image: np.ndarray, image_name: str = None) -> Dict:
        """
        Run inference on an input image.
        
        Args:
            image: Input image as numpy array (RGB format)
            image_name: Optional name of the image (not used for folder naming, kept for compatibility)
            
        Returns:
            Dictionary with inference results and output paths
        """
        # Create unique timestamp with seconds and microseconds
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Remove last 3 digits of microseconds
        
        # Create output folder with just the timestamp (only if saving is enabled)
        output_dir = os.path.join(self.output_base_dir, f"yoloe_{timestamp}")
        
        if self.save_output_files:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nRunning inference on image with shape {image.shape}...")
        
        # Run YOLOE prediction
        results = self.model.predict(image, save=False)
        
        # Extract results
        categories = set()
        bboxes_data = []
        masks_list = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Get detections
                class_ids = result.boxes.cls.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                confidences = result.boxes.conf.cpu().numpy()
                
                # Get masks if available - properly scaled to original image
                if result.masks is not None:
                    # Get original image shape from result
                    orig_shape = result.orig_shape  # (height, width)
                    
                    # Use the masks in the original image space
                    # result.masks.xy contains polygon coordinates in original image space
                    # But we need the actual mask arrays, so we'll manually scale them
                    masks_model_space = result.masks.data.cpu().numpy()  # Shape: (N, 640, 640)
                    
                    # Scale masks to original image size
                    # We need to account for letterbox padding
                    masks_orig = []
                    for mask_640 in masks_model_space:
                        # Scale mask from 640x640 to original image size
                        # This needs to account for the letterbox transformation
                        mask_orig = self._scale_mask_to_original(
                            mask_640, 
                            orig_shape,
                            model_input_size=(640, 640)
                        )
                        masks_orig.append(mask_orig)
                else:
                    masks_orig = None
                    orig_shape = None
                
                # Process each detection
                for idx, class_id in enumerate(class_ids):
                    label = self.names[int(class_id)]
                    categories.add(label)
                    
                    # Store bounding box information
                    bbox_info = {
                        "instance_id": idx,
                        "category": label,
                        "confidence": float(confidences[idx]),
                        "bbox": {
                            "x1": float(boxes[idx][0]),
                            "y1": float(boxes[idx][1]),
                            "x2": float(boxes[idx][2]),
                            "y2": float(boxes[idx][3])
                        }
                    }
                    bboxes_data.append(bbox_info)
                    
                    # Store mask if available
                    if masks_orig is not None and idx < len(masks_orig):
                        masks_list.append({
                            "mask": masks_orig[idx],
                            "label": label,
                            "instance_id": idx,
                            "orig_shape": orig_shape
                        })
        
        # Save files only if enabled
        categories_file = None
        bboxes_file = None
        
        if self.save_output_files:
            # Save categories
            categories_file = os.path.join(output_dir, "categories.json")
            with open(categories_file, 'w') as f:
                json.dump(sorted(list(categories)), f, indent=2)
            print(f"Saved categories to: {categories_file}")
            
            # Save bounding boxes
            bboxes_file = os.path.join(output_dir, "boundingboxes.json")
            with open(bboxes_file, 'w') as f:
                json.dump(bboxes_data, f, indent=2)
            print(f"Saved bounding boxes to: {bboxes_file}")
            
            # Save highlighted mask images
            if masks_list:
                self._save_mask_highlighted_images(image, masks_list, output_dir)
        else:
            print(f"Output file saving is disabled (save_output_files=False)")
        
        return {
            "timestamp": timestamp,
            "output_dir": output_dir,
            "num_detections": len(bboxes_data),
            "categories": sorted(list(categories)),
            "categories_file": categories_file,
            "bboxes_file": bboxes_file,
            "bboxes_data": bboxes_data,  # For ROS2 usage
            "masks_list": masks_list  # For ROS2 usage
        }
    
    def _save_mask_highlighted_images(self, image: np.ndarray, masks_list: List[Dict], 
                                      output_dir: str) -> None:
        """
        Save highlighted images with masks overlayed.
        
        Args:
            image: Original image (RGB format)
            masks_list: List of dictionaries with mask, label, instance_id, and orig_shape
            output_dir: Directory to save output images
        """
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Orange color for highlighting (light orange)
        highlight_color = (0, 165, 255)  # BGR format: (B, G, R) = (0, 165, 255) = light orange
        alpha = 0.5
        
        for mask_data in masks_list:
            mask = mask_data["mask"]
            label = mask_data["label"]
            instance_id = mask_data["instance_id"]
            
            # Ensure mask is in the correct format and size
            if mask.shape[0] != image_bgr.shape[0] or mask.shape[1] != image_bgr.shape[1]:
                # Resize if needed (should rarely happen now with letterbox correction)
                target_h, target_w = image_bgr.shape[0], image_bgr.shape[1]
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (target_w, target_h),  # (width, height)
                    interpolation=cv2.INTER_LINEAR
                )
                mask_resized = (mask_resized > 0.5).astype(np.uint8)
            else:
                mask_resized = (mask > 0.5).astype(np.uint8)
            
            # Create highlighted image
            highlighted_image = self._highlight_image_mask(
                image_bgr.copy(), 
                mask_resized, 
                label=label, 
                color=highlight_color, 
                alpha=alpha
            )
            
            # Save image
            output_path = os.path.join(output_dir, f"mask_{instance_id}_highlighted.png")
            cv2.imwrite(output_path, highlighted_image)
            print(f"Saved highlighted mask to: {output_path}")
    
    def _highlight_image_mask(self, image: np.ndarray, mask: np.ndarray, 
                             label: str = "unknown", color: Tuple[int, int, int] = (0, 255, 0), 
                             alpha: float = 0.5, fixed_width: int = 800) -> np.ndarray:
        """
        Overlay a segmentation mask on the image and add a label.
        Similar to TALOS implementation.
        
        Args:
            image: Input image in BGR format
            mask: Binary mask
            label: Text label to display
            color: Color for mask overlay in BGR format
            alpha: Transparency factor for overlay
            fixed_width: Width to resize image to (maintains aspect ratio)
            
        Returns:
            Image with highlighted mask and label
        """
        # Resize the image to a fixed width, keeping aspect ratio
        aspect_ratio = image.shape[1] / float(image.shape[0])
        new_height = int(fixed_width / aspect_ratio)
        resized_image = cv2.resize(image, (fixed_width, new_height))
        
        # Resize the mask to match the resized image
        # cv2.resize expects (width, height)
        resized_mask = cv2.resize(
            mask.astype(np.float32), 
            (fixed_width, new_height),  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        # Threshold back to binary
        resized_mask = (resized_mask > 0.5).astype(np.uint8)
        
        # Create mask overlay
        mask_overlay = np.zeros_like(resized_image, dtype=np.uint8)
        mask_bool = resized_mask.astype(bool)
        mask_overlay[mask_bool] = color
        
        # Blend original image with mask overlay
        overlayed_image = cv2.addWeighted(resized_image, 1 - alpha, mask_overlay, alpha, 0)
        
        # Add label text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x, text_y = 10, 10 + text_size[1]  # Position in top-left corner
        
        # Text background (black rectangle)
        cv2.rectangle(
            overlayed_image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        # Overlay text
        cv2.putText(overlayed_image, label, (text_x, text_y), font, font_scale, color, thickness)
        
        return overlayed_image
