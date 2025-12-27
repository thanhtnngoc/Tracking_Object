#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2
import math
from ultralytics import YOLO
from threading import Thread
import sys
from collections import deque
import numpy as np

class VideoStream_UsbCam:
    """USB video stream running in a separate thread."""
    def __init__(self, cam_index=2, resolution=(640, 480)):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open USB camera!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        """Continuously update frames in the background."""
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        """Return the latest frame."""
        return self.ret, self.frame

    def stop(self):
        """Stop the video stream and release the camera."""
        self.stopped = True
        self.cap.release()


class ObjectScanner(Node):
    """ROS2 node for person detection using YOLOv8 (built-in bounding boxes)."""

    def __init__(self, show_display=False):
        super().__init__('yolo_person_scanner')
        self.get_logger().info("YOLOv8 Person Scanner node started")

        self.show_display = show_display

        self.stream = VideoStream_UsbCam()

        self.model = YOLO("yolov8n.pt") 

        self.publisher = self.create_publisher(Int32MultiArray, '/yolo_object_result', 10)

        self.timer = self.create_timer(0.03, self.timer_callback)

        self.box_history = deque(maxlen=5)

    def display_frame(self, frame):
        """Display frame only if enabled."""
        if not self.show_display:
            return 
        cv2.imshow("YOLOv8 Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            self.get_logger().info("ESC pressed — shutting down gracefully...")
            self.stream.stop()               
            cv2.destroyAllWindows()    
            rclpy.shutdown()          
            sys.exit(0)
            return    

    def timer_callback(self):
        """Main callback for detection and publishing results."""
        ret, frame = self.stream.read()
        if not ret or frame is None:
            print("No frame received from camera")
            return
        
        frame_resized = cv2.resize(frame, (640, 480)) 

        h, w, _ = frame_resized.shape
        cx_frame = w // 2
        cy_frame = h // 2

        results = self.model(frame_resized, classes=[0], conf=0.6, verbose=False)
        boxes = results[0].boxes

        annotated_frame = results[0].plot()
        cv2.circle(annotated_frame, (cx_frame, cy_frame), 6, (255, 0, 0), -1) 

        detection_flag = 0
        cx = cy = 0
        w_box = h_box = 0 
        dx = dy = 0
        theta_deg = 0

        distances = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                dx = cx - cx_frame  # HPDist
                dy = cy - cy_frame  # VPDist

                theta_rad = math.atan2(dx, -dy)  
                theta_deg = math.degrees(theta_rad) 
                
                dist = math.sqrt(dx**2 + dy**2)
                distances.append((dist, dx, dy, (cx, cy)))

            dist_val, dx, dy, (cx, cy) = min(distances, key=lambda x: x[0])
            detection_flag = 1
            if detection_flag == 1:
                self.box_history.append((cx, cy, w_box, h_box))

            # Nếu đủ 5 khung → tính trung bình
            if len(self.box_history) >= 10:
                arr = np.array(self.box_history)
                cx, cy, w_box, h_box = np.mean(arr, axis=0).astype(int)

            if self.show_display:
                cv2.rectangle(annotated_frame,
                  (cx - w_box//2, cy - h_box//2),
                  (cx + w_box//2, cy + h_box//2),
                  (0, 255, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.line(annotated_frame, (cx_frame, cy_frame), (cx, cy), (0, 0, 255), 2)
                cv2.line(annotated_frame, (cx_frame, cy_frame), (cx_frame, cy), (255, 0, 0), 2)
                cv2.line(annotated_frame, (cx_frame, cy), (cx, cy), (0, 200, 50), 2)
                cv2.putText(annotated_frame, f"u={cx}, v={cy}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            self.box_history.clear()
            detection_flag = 0

        cx, cy, w_box, h_box = map(int, (cx, cy, w_box, h_box))

        msg = Int32MultiArray()
        msg.data = [cx, cy, w, h, detection_flag]
        self.publisher.publish(msg)
        self.get_logger().info(f"u = {cx}, v = {cy}, dx = {dx}, dy = {dy}, w = {w}, h = {h}, theta = {theta_deg}, flag = {detection_flag}")

        self.display_frame(annotated_frame)

def main(args=None):
    """Initialize and spin the ROS2 node."""
    rclpy.init(args=args)
    node = ObjectScanner(show_display=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
