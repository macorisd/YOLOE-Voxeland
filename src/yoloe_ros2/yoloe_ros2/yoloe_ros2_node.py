import numpy as np
import cv2

import rclpy
import rclpy.node
import sensor_msgs.msg
from cv_bridge import CvBridge

from segmentation_msgs.srv import SegmentImage
from segmentation_msgs.msg import SemanticInstance2D
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose

from yoloe_ros2.yoloe_ros2_main import YOLOERos2


class YOLOERos2Node(rclpy.node.Node):
    def __init__(self):
        super().__init__('yoloe_ros2_node')
        
        self.publish_visualization = self.declare_parameter("publish_visualization", True).value 
        visualization_topic = self.declare_parameter("visualization_topic", "/yoloe/segmentedImage").value
        self.visualization_pub = self.create_publisher(sensor_msgs.msg.Image, visualization_topic, 1)

        self.cv_bridge = CvBridge()

        self.segment_image_srv = self.create_service(SegmentImage, "/yoloe/segment", self.segment_image)
        self.yoloe_pipeline = YOLOERos2()
        
        self._logger.info("Done setting up!")
        self._logger.info(f"Advertising service: {self.segment_image_srv.srv_name}")

    def segment_image(self, request, response):
        """
        Service callback to segment an image using YOLOE.
        
        Args:
            request: SegmentImage service request with an image
            response: SegmentImage service response with semantic instances
            
        Returns:
            response with detected instances
        """
        # Convert ROS Image message to numpy array (BGR)
        numpy_image = self.cv_bridge.imgmsg_to_cv2(request.image)

        # Run YOLOE pipeline
        results, total_time = self.yoloe_pipeline.run(input_image=numpy_image)
        
        self._logger.info(f"YOLOE inference completed in {total_time:.2f} seconds")
        
        # Extract results from YOLOE output
        bboxes_data = results.get("bboxes_data", [])
        masks_list = results.get("masks_list", [])
        
        # Process each detection
        for bbox_info in bboxes_data:
            semantic_instance = SemanticInstance2D()

            instance_id = bbox_info.get("instance_id")
            label = bbox_info.get("category", "unknown")
            confidence = bbox_info.get("confidence", 0.01)
            bbox = bbox_info.get("bbox", {})
            
            x_min = bbox.get("x1", 0)
            y_min = bbox.get("y1", 0)
            x_max = bbox.get("x2", 0)
            y_max = bbox.get("y2", 0)
            
            width = x_max - x_min
            height = y_max - y_min

            # Find corresponding mask
            mask = None
            for mask_data in masks_list:
                if mask_data.get("instance_id") == instance_id:
                    mask = mask_data.get("mask")
                    break
            
            # Convert mask to ROS Image message if available
            if mask is not None:
                # Ensure mask is in correct format (uint8, 0-255)
                if mask.dtype != np.uint8:
                    mask_uint8 = (mask > 0.5).astype("uint8") * 255
                else:
                    mask_uint8 = mask
                semantic_instance.mask = self.cv_bridge.cv2_to_imgmsg(mask_uint8)
            else:
                # Create empty mask if not available
                empty_mask = np.zeros((numpy_image.shape[0], numpy_image.shape[1]), dtype=np.uint8)
                semantic_instance.mask = self.cv_bridge.cv2_to_imgmsg(empty_mask)

            # Create Detection2D message
            detection = Detection2D()
            detection.bbox = BoundingBox2D()
            detection.bbox.center.position.x = x_min + width / 2.0
            detection.bbox.center.position.y = y_min + height / 2.0
            detection.bbox.size_x = width
            detection.bbox.size_y = height

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = confidence
            detection.results.append(hypothesis)

            semantic_instance.detection = detection
            response.instances.append(semantic_instance)

            self._logger.info(f"Detected {label} with score {confidence:.2f} at bbox [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]")

        # Visualization
        if self.publish_visualization:
            img_vis = numpy_image.copy()
            rng = np.random.default_rng(seed=42)
            
            for bbox_info in bboxes_data:
                bbox = bbox_info.get("bbox", {})
                label = bbox_info.get("category", "unknown")
                instance_id = bbox_info.get("instance_id")
                
                x_min = int(bbox.get("x1", 0))
                y_min = int(bbox.get("y1", 0))
                x_max = int(bbox.get("x2", 0))
                y_max = int(bbox.get("y2", 0))

                # Draw red bounding box and label
                cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(img_vis, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Generate a random color for the mask (in BGR)
                random_color = rng.integers(0, 256, size=3, dtype=np.uint8)

                # Find and blend the mask with the random color
                for mask_data in masks_list:
                    if mask_data.get("instance_id") == instance_id:
                        mask = mask_data.get("mask")
                        if mask is not None:
                            mask_bool = mask > 0.5
                            img_vis[mask_bool] = (
                                img_vis[mask_bool] * 0.5 + random_color * 0.5
                            ).astype(np.uint8)
                        break

            # Convert from BGR to RGB before publishing
            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            image_msg = self.cv_bridge.cv2_to_imgmsg(img_vis_rgb, encoding="rgb8")
            self.visualization_pub.publish(image_msg)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = YOLOERos2Node()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
