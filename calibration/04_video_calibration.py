import cv2 as cv
import numpy as np
import pickle
import os

# 카메라 파라미터를 바탕으로 동영상 왜곡 보정

param_file = os.path.join("calibration_result", "calibration.pkl")
input_video_path = "c_103.mp4"  # 동영상 파일 경로

# -----------------------------
# 1. 카메라 파라미터 불러오기
# -----------------------------

if not os.path.exists(param_file):
    print(f"에러: {param_file} 파일을 찾을 수 없습니다.")
    exit()

with open(param_file, "rb") as f:
    cameraMatrix, distCoeffs = pickle.load(f)

print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coeffs:\n", distCoeffs)

# -----------------------------
# 2. 입력 동영상 파일 열기
# -----------------------------

cap = cv.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: 입력 동영상을 열 수 없습니다.")
    exit()

# 동영상의 프레임 크기와 FPS 확인
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
print(f"입력 동영상 해상도: {frame_width} x {frame_height}, FPS: {fps}")

# -----------------------------
# 3. 새로운 카메라 행렬 & ROI 계산
# -----------------------------
# alpha=1 : 원본 전체 보존하려다 보면 가장자리 왜곡이 심해질 수 있음
# alpha=0 : 불필요한 부분을 잘라내면서 왜곡을 최소화
alpha = 0.0  # 필요에 따라 0.0~1.0 사이 값을 시도
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, distCoeffs, (frame_width, frame_height), alpha, (frame_width, frame_height)
)

x, y, w_roi, h_roi = roi
print(f"ROI: x={x}, y={y}, w={w_roi}, h={h_roi}")

# -----------------------------
# 4. 출력 동영상 설정 (VideoWriter)
# -----------------------------
output_video_path = "undistorted_" + input_video_path + ".mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
# 여기서는 원본 크기로 저장(ROI 크롭을 적용하지 않으면 frame_width x frame_height)
# 만약 ROI를 적용해 잘라낸 크기로 저장하고 싶다면 (w_roi, h_roi) 사용
out_size = (frame_width, frame_height)
out = cv.VideoWriter(output_video_path, fourcc, fps, out_size)

# -----------------------------
# 5. 동영상 프레임 처리
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # (1) undistort 로 테스트 하는 경우 - 가운데 영상
    undistorted_simple = cv.undistort(frame, cameraMatrix, distCoeffs)

    # (2) getOptimalNewCameraMatrix 활용 보정 - 오른쪽 영상
    undistorted_optimal = cv.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)

    # ROI 크롭을 적용해보고 싶다면 주석 해제
    # -> 왜곡이 줄어든 핵심 영역만 사용하게됨
    # undistorted_optimal = undistorted_optimal[y:y+h_roi, x:x+w_roi]

    # -----------------------------
    # 6. 결과 화면 표시 (비교)
    # -----------------------------
    # 세 영상을 가로로 이어붙여 확인
    # 왼쪽 - 중앙 - 오른쪽
    # 원본 | (1) | (2)
    show_w, show_h = 480, 270
    frame_resized = cv.resize(frame, (show_w, show_h))
    simple_resized = cv.resize(undistorted_simple, (show_w, show_h)) # (1)
    optimal_resized = cv.resize(undistorted_optimal, (show_w, show_h)) # (2)

    combined = cv.hconcat([frame_resized, simple_resized, optimal_resized])
    cv.imshow("Original | Simple Undistort | Optimal + ROI", combined)

    # 최종적으로 저장할 보정 영상을 undistorted_optimal로 결정
    # (필요에 따라 simple undistort 결과를 저장해도 됨)
    # 만약 ROI 크롭을 했으면, out_size도 (w_roi, h_roi)로 맞춰야 함
    # 여기서는 원본 크기로 저장 -> 크롭 안 했으므로 그대로 사용
    final_frame = undistorted_optimal

    # 원본 해상도로 맞추기(혹은 크롭 상태면 크롭 크기로)
    final_frame_resized = cv.resize(final_frame, out_size)
    out.write(final_frame_resized)

    # ESC 키(27) 누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()

print(f"보정된 동영상이 '{output_video_path}'에 저장되었습니다.")
