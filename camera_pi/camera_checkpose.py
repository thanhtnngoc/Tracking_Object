import cv2
import numpy as np
import math

def rotationVectorToEulerAngles(rvec):
    # Chuyển rvec sang ma trận xoay
    R, _ = cv2.Rodrigues(rvec)
    
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # Chuyển sang độ
    return np.degrees([x, y, z])  # roll, pitch, yaw
# Thông số đã biết từ calibration
with np.load("camera_params.npz") as X:
    mtx, dist = X['mtx'], X['dist']

# Kích thước checkerboard: số inner corners
checkerboard_size = (8, 5)
square_size = 30  # kích thước ô vuông (mét, cm, tùy bạn đặt)

# Tạo object points 3D chuẩn (0,0,0), (1,0,0), (2,0,0), ... theo kích thước square_size
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Đọc ảnh
img = cv2.imread("images_calib/img_000.jpg")
h, w = img.shape[:2]

# Tính toán ma trận hiệu chỉnh
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# Undistort
undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
# Cắt ảnh nếu cần
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tìm góc checkerboard
ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

if ret:
    # Làm mịn góc
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    
    # Tính pose
    retval, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
     # Lấy góc quay dạng độ
    euler_angles = rotationVectorToEulerAngles(rvec)  # [roll, pitch, yaw]
    roll, pitch, yaw = euler_angles*180/3.14
    print("Rotation vector:\n", rvec)
    print("Translation vector:\n", tvec)
    text = f"Roll: {roll:.1f} deg, Pitch: {pitch:.1f} deg, Yaw: {yaw:.1f} deg"
    print(text)
    # cv2.putText(undistorted_img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    # Vẽ các góc tìm được lên ảnh
    # cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
    # cv2.imshow("Pose Estimation", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("Không tìm thấy góc checkerboard trong ảnh.")