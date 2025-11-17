#!/usr/bin/env python3
import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from threading import Thread

class SubscriberObject(Node):
    def __init__(self):
        super().__init__('subscriber_object')
        self.subscription = self.create_subscription(Int32MultiArray,'/yolo_object_result', self.listener_callback, 10)
        self.get_logger().info("Object scanner started, waiting for messages...")

    def listener_callback(self, msg):
        dx, dy = msg.data
        self.get_logger().info(f"Nhận từ YOLO: dx = {dx}, dy = {dy}")

def main():
    rclpy.init()
    node = SubscriberObject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()