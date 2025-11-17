import cv2

# Mở camera USB (thường là /dev/video0)
cap = cv2.VideoCapture(0)  # 0 = camera đầu tiên, đổi thành 1 nếu bạn có nhiều camera

if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình!")
        break

    # Hiển thị hình ảnh
    cv2.imshow("USB Camera", frame)

    # Nhấn ESC để thoát
    key = cv2.waitKey(1)
    if key == 27:  # 27 là mã ASCII của phím ESC
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()