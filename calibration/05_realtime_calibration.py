import cv2 as cv
import numpy as np
import pickle
import os

# "calibration_result" 폴더 내에 "calibration.pkl" 파일이 있다고 가정
param_file = os.path.join("calibration_result", "calibration.pkl")
CAMERA103 = "rtsp://admin:asdf1346@@192.168.10.103/stream1/out.h264"

# -----------------------------
# 1. 카메라 파라미터 불러오기
# -----------------------------
if not os.path.exists(param_file):
    print(f"에러: {param_file} 파일을 찾을 수 없습니다.")
    exit()

# cameraMatrix, distCoeffs 불러오기
with open(param_file, "rb") as f:
    cameraMatrix, distCoeffs = pickle.load(f)

print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coeffs:\n", distCoeffs)

# -----------------------------
# 2. 웹캠 열기
# -----------------------------

cap = cv.VideoCapture(CAMERA103)
if not cap.isOpened():
    print("에러: 웹캠을 열 수 없습니다.")
    exit()

# -----------------------------
# 3. 실시간 영상 왜곡 보정
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("에러: 영상 프레임을 가져오지 못했습니다.")
        break

    # (1) 간단한 undistort 사용
    undistorted_simple = cv.undistort(frame, cameraMatrix, distCoeffs)

    # (2) getOptimalNewCameraMatrix 사용 (더 나은 결과를 위해)
    # 프레임 크기 구하기
    h, w = frame.shape[:2]

    # 최적화된 새 카메라 행렬과 ROI 정보 구하기
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (w, h), 1, (w, h)
    )
    undistorted_optimal = cv.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)

    # ROI 영역만 잘라내기 (optical center 등 고려)
    x, y, w_roi, h_roi = roi
    undistorted_optimal = undistorted_optimal[y : y + h_roi, x : x + w_roi]

    # -----------------------------
    # 4. 결과 화면 표시
    # -----------------------------
    # 원본과 보정 영상을 나란히 보기 위해 크기를 맞춤
    # (화면 크기에 따라 조정)
    resized_original = cv.resize(frame, (640, 360))
    resized_undistort = cv.resize(undistorted_optimal, (640, 360))

    # 두 이미지를 가로로 연결
    combined = cv.hconcat([resized_original, resized_undistort])

    cv.imshow("Original (Left) vs Undistorted (Right)", combined)

    # ESC 키(27) 누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
