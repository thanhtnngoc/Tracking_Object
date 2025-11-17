import cv2
import numpy as np
import glob
import os

# ⚙️ Cấu hình kích thước checkerboard (số ô vuông bên trong)
CHECKERBOARD = (8, 5)  # Số góc bên trong, không phải số ô!
square_size = 30  # Kích thước 1 ô vuông thực tế (nếu cần đơn vị, ví dụ mm)

# 🔹 Tạo mảng điểm object trong không gian 3D (z=0 vì checkerboard phẳng)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 🔹 Danh sách lưu điểm object và image
objpoints = []  # 3D point trong thế giới thực
imgpoints = []  # 2D point trong ảnh

# 🔍 Load ảnh checkerboard đã chụp
dir = "/home/thanh/ros2_ws/src/p_detect_object/images_calib"
images = glob.glob(os.path.join(dir, "*.jpg"))

print(f"Tìm thấy {len(images)} ảnh")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm góc checkerboard
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    print(f"{fname} - Góc tìm thấy: {ret}")

    if ret:
        # Tăng độ chính xác tọa độ góc
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # Hiển thị ảnh đã dò góc
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 🎯 Hiệu chỉnh camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 📄 In kết quả
print("✅ Calibration successful!")
print("Camera matrix (mtx):\n", mtx)
print("Distortion coefficients (dist):\n", dist)
print("Rotation vectors (rvecs):\n", rvecs)
print("Translation vectors (tvecs):\n", tvecs)


save_dir = "/home/thanh/ros2_ws/src/p_detect_object/"
os.makedirs(save_dir, exist_ok=True)  # tạo nếu chưa có
save_path = os.path.join(save_dir, "usbcamera_params.npz")

np.savez(save_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print(f"✅ Đã lưu file: {save_path}")