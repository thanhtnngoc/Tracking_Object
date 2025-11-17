# hello Nam hehe
import cv2
import numpy as np
from threading import Thread 
from picamera2 import Picamera2

class VideoStream:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (1640, 1232)}
            )
        )
        self.picam2.start()
        self.frame = self.picam2.capture_array()
        self.stopped = False
        Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array()

    def read(self):
        return True, self.frame

    def stop(self):
        self.stopped = True
        self.picam2.stop()

# Load calibration results 
with np.load("camera_params.npz") as X:
    mtx, dist = X['mtx'], X['dist']

stream = VideoStream()
while True:
    ret, img = stream.read()
    if not ret:
        print("Không đọc được frame, thoát")
        break
    h, w = img.shape[:2]

    # Tính toán ma trận hiệu chỉnh
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # Cắt ảnh nếu cần
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    cv2.imshow("Undistorted", undistorted_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()

