import sys
import cv2
import numpy as np
import os
import pickle

# =====================================
# 1. 파일 경로 및 출력 파일 설정
# =====================================
video1_name = 'videos/test_left'
video2_name = 'videos/test_rigth'
extension = '.mp4'
video1_path = video1_name + extension  # 왼쪽 카메라 영상
video2_path = video2_name + extension  # 오른쪽 카메라 영상

# 출력 디렉토리 지정 및 생성
output_dir = "output_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 최종 스티칭 결과 파일
stitched_video_path = os.path.join(
    output_dir, f"{os.path.basename(video1_name)}_{os.path.basename(video2_name)}_stitched2.mp4"
)

# 캘리브레이션 파일
left_calib_file = "calibration.pkl"         # 왼쪽 카메라
right_calib_file = "calibration_104.pkl"      # 오른쪽 카메라

# =====================================
# 2. 비디오 캡처 및 기본 정보
# =====================================
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap1.get(cv2.CAP_PROP_FPS)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
if not (ret1 and ret2):
    print("Cannot read initial frames.")
    sys.exit(1)

# =====================================
# 3. 캘리브레이션 데이터 불러오기
# =====================================
if not os.path.exists(left_calib_file) or not os.path.exists(right_calib_file):
    print("Calibration file(s) not found.")
    sys.exit(1)

with open(left_calib_file, "rb") as f:
    left_data = pickle.load(f)
    K_left, dist_left = left_data[0], left_data[1]

with open(right_calib_file, "rb") as f:
    right_data = pickle.load(f)
    K_right, dist_right = right_data[0], right_data[1]

# =====================================
# 4. 보정 함수 정의
# =====================================
def undistort_frame(frame, K, dist_coeffs):
    return cv2.undistort(frame, K, dist_coeffs)

# =====================================
# 5. SIFT, FLANN, CLAHE 객체 생성
# =====================================
sift = cv2.SIFT_create(nfeatures=12000)
flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# =====================================
# 6. 최적의 호모그래피 매트릭스 계산 (첫 N 프레임에서)
# =====================================
best_inlier_count = -1
best_H = None
num_frames_to_check = 50  # 예를 들어 처음 50프레임을 사용

# 비디오 위치 재설정
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0
while frame_idx < num_frames_to_check:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        break

    # 보정 적용
    frame1_undistorted = undistort_frame(frame1, K_left, dist_left)
    frame2_undistorted = undistort_frame(frame2, K_right, dist_right)

    # 그레이스케일 및 CLAHE 적용
    gray1 = cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2GRAY)
    gray1 = clahe.apply(gray1)
    gray2 = clahe.apply(gray2)

    # SIFT 특징점 검출 및 기술자 계산
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        frame_idx += 1
        continue

    # FLANN 매칭 및 ratio test 적용
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(good_matches) < 4:
        frame_idx += 1
        continue

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # 호모그래피 계산 (pts2를 pts1에 맞춤)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        frame_idx += 1
        continue

    inlier_count = int(np.sum(mask))
    if inlier_count > best_inlier_count:
        best_inlier_count = inlier_count
        best_H = H

    frame_idx += 1

if best_H is None:
    print("최적의 호모그래피 매트릭스를 계산하지 못했습니다.")
    sys.exit(1)
else:
    print("최적의 호모그래피 매트릭스 선택됨. Inliers:", best_inlier_count)

# =====================================
# 7. 최종 스티칭을 위한 VideoWriter 설정
# =====================================
# 두 영상의 너비를 합친 크기로 결과 영상 생성
out_w = frame_width * 2
out_h = frame_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_stitched = cv2.VideoWriter(stitched_video_path, fourcc, fps, (out_w, out_h))
if not out_stitched.isOpened():
    print("Error: VideoWriter could not be opened.")
    sys.exit(1)

# =====================================
# 8. 최적의 호모그래피 매트릭스를 사용하여 전체 영상 스티칭
# =====================================
# 비디오 위치 재설정
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("스티칭 작업 시작...")
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        break

    # 각 프레임에 대해 보정 적용
    frame1_undistorted = undistort_frame(frame1, K_left, dist_left)
    frame2_undistorted = undistort_frame(frame2, K_right, dist_right)

    # 왼쪽 영상은 그대로 사용하고, 오른쪽 영상에 최적 호모그래피 적용
    warped_frame2 = cv2.warpPerspective(frame2_undistorted, best_H, (out_w, out_h))

    # 스티칭 기본 베이스 생성: 좌측 영상과 오른쪽 영상을 이어붙임
    stitched_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    stitched_frame[:, :frame_width] = frame1_undistorted
    stitched_frame[:, frame_width:] = warped_frame2[:, frame_width:]

    # (옵션) 블렌딩 등을 추가하여 경계 부분을 부드럽게 처리할 수 있음

    out_stitched.write(stitched_frame)

cap1.release()
cap2.release()
out_stitched.release()

print("스티칭 동영상 저장 완료:", stitched_video_path)
