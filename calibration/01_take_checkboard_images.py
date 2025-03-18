import cv2

# Taking checkerboard images for camera calibration
# 카메라 교정용 체커보드 이미지 촬영

# Camera which you want to calibrate
CAMERA103 = "rtsp://admin:asdf1346@@192.168.10.103/stream1/out.h264"
CAMERA104 = "rtsp://admin:asdf1346@@192.168.10.104/stream1/out.h264"

cap = cv2.VideoCapture(CAMERA104)
num = 0
while cap.isOpened():
    succes, img = cap.read()
    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('../images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1
    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()