import sys
import cv2
import numpy as np
import os
import pickle

# ================================
# 1. 파일 경로 및 출력 파일 설정
# ================================
video1_name = 'left'
video2_name = 'rigth'
extension = '.mp4'
video1_path = video1_name + extension # 왼쪽 카메라 영상
video2_path = video2_name + extension  # 오른쪽 카메라 영상

# 1) 일반 스티칭 결과 파일
stitched_video_path = f"{video1_name}_{video2_name}_stitched.mp4"
# 2) 색상 오버레이 스티칭 결과 파일
overlay_video_path = f"{video1_name}_{video2_name}_stitched_overlay.mp4"
# 3) 크롭된 오버레이 결과 파일
cropped_overlay_video_path = f"{video1_name}_{video2_name}_stiched_cropped.mp4"

# 카메라 파라미터 (pickle) 파일
left_calib_file = "calibration.pkl"   # 왼쪽 카메라
right_calib_file = "calibration.pkl"  # 오른쪽 카메라

# ================================
# 2. 옵션 설정 (True/False)
# ================================
want_normal_stitched = True     # 일반 스티칭 영상 생성 여부
want_overlay_stitched = False    # 색상 오버레이 영상 생성 여부
want_crop_overlay = True        # 색상 오버레이 영상에 대해 크롭 수행 여부

# ================================
# 3. 비디오 캡처 및 기본 정보
# ================================
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

# ================================
# 4. 저장된 캘리브레이션 데이터 불러오기
# ================================
if not os.path.exists(left_calib_file):
    print("Error: Left calibration file not found:", left_calib_file)
    sys.exit(1)
if not os.path.exists(right_calib_file):
    print("Error: Right calibration file not found:", right_calib_file)
    sys.exit(1)

with open(left_calib_file, "rb") as f:
    left_data = pickle.load(f)  # (cameraMatrix, distCoeffs)
    K_left, dist_left = left_data[0], left_data[1]

with open(right_calib_file, "rb") as f:
    right_data = pickle.load(f)
    K_right, dist_right = right_data[0], right_data[1]

print("Left Camera Matrix:\n", K_left)
print("Left Distortion Coeffs:\n", dist_left)
print("Right Camera Matrix:\n", K_right)
print("Right Distortion Coeffs:\n", dist_right)

# ================================
# 5. 보정 함수 정의
# ================================
def undistort_frame(frame, K, dist_coeffs):
    return cv2.undistort(frame, K, dist_coeffs)

# ================================
# 6. 초기 프레임 보정 및 대비 향상
# ================================
frame1_undistorted = undistort_frame(frame1, K_left, dist_left)
frame2_undistorted = undistort_frame(frame2, K_right, dist_right)

gray1 = cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray1 = clahe.apply(gray1)
gray2 = clahe.apply(gray2)

# ================================
# 7. 특징점 검출 및 매칭 (SIFT + FLANN)
# ================================
sift = cv2.SIFT_create(nfeatures=12000)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
matches = flann.knnMatch(des1, des2, k=2)

good_matches, pts1, pts2 = [], [], []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
        good_matches.append(m)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# ================================
# 8. 기하학적 관계 추정 (Fundamental / Homography)
# ================================
F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 2.0, 0.99)
pts1_inliers = pts1[mask_f.ravel() == 1]
pts2_inliers = pts2[mask_f.ravel() == 1]

# 두 번째 영상을 첫 번째 영상 기준으로 정렬하기 위한 Homography
H, _ = cv2.findHomography(pts2_inliers, pts1_inliers, cv2.RANSAC, 4.0)

# ================================
# 9. 비디오 재생 위치 초기화
# ================================
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ================================
# 10. 출력 비디오 설정
# ================================
out_w, out_h = frame_width * 2, frame_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# (1) 일반 스티칭 VideoWriter
if want_normal_stitched:
    out_stitched = cv2.VideoWriter(stitched_video_path, fourcc, fps, (out_w, out_h))
else:
    out_stitched = None

# (2) 색상 오버레이 스티칭 VideoWriter
if want_overlay_stitched:
    out_overlay = cv2.VideoWriter(overlay_video_path, fourcc, fps, (out_w, out_h))
else:
    out_overlay = None

# ================================
# 11. Feather 블렌딩용 마스크
# ================================
def create_feather_mask(width, height, overlap_width):
    mask = np.ones((height, width), dtype=np.float32)
    for i in range(overlap_width):
        alpha = i / overlap_width
        mask[:, width - overlap_width + i] = 1 - alpha
    return cv2.merge([mask, mask, mask])

overlap_width = 150
print("Stitching video frames...")

# ================================
# 12. 색상 오버레이 함수
# ================================
def apply_color_overlay(stitched_frame, frame_width, overlap_width):
    """
    왼쪽(빨강), 겹치는 영역(파랑), 오른쪽(초록) 영역을 표시
    alpha=0.3 정도로 영상과 합성
    """
    alpha = 0.3
    
    # 왼쪽 영역(빨강)
    overlay_left = np.zeros_like(stitched_frame, dtype=np.uint8)
    overlay_left[:, :frame_width] = (0, 0, 255)
    stitched_frame = cv2.addWeighted(overlay_left, alpha, stitched_frame, 1 - alpha, 0)
    
    # 겹치는 영역(파랑)
    overlay_overlap = np.zeros_like(stitched_frame, dtype=np.uint8)
    overlay_overlap[:, frame_width - overlap_width:frame_width] = (255, 0, 0)
    stitched_frame = cv2.addWeighted(overlay_overlap, alpha, stitched_frame, 1 - alpha, 0)
    
    # 오른쪽 영역(초록)
    overlay_right = np.zeros_like(stitched_frame, dtype=np.uint8)
    overlay_right[:, frame_width:] = (0, 255, 0)
    stitched_frame = cv2.addWeighted(overlay_right, alpha, stitched_frame, 1 - alpha, 0)
    
    return stitched_frame

# ================================
# 13. 스티칭 루프
# ================================
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        break
    
    # (1) 각 프레임 보정
    frame1_undistorted = undistort_frame(frame1, K_left, dist_left)
    frame2_undistorted = undistort_frame(frame2, K_right, dist_right)
    
    # (2) Homography로 두 번째 영상 워프
    warped_frame2 = cv2.warpPerspective(frame2_undistorted, H, (out_w, out_h))
    
    # (3) 스티칭 기본 베이스
    stitched_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    stitched_frame[:, :frame_width] = frame1_undistorted
    
    # (4) Feather 블렌딩
    mask1 = create_feather_mask(frame_width, frame_height, overlap_width)
    mask2 = 1 - mask1
    
    left_overlap = stitched_frame[:, frame_width - overlap_width:frame_width].astype(np.float32)
    right_overlap = warped_frame2[:, frame_width - overlap_width:frame_width].astype(np.float32)
    blended_region = (left_overlap * mask1[:, -overlap_width:] +
                      right_overlap * mask2[:, frame_width - overlap_width:frame_width])
    stitched_frame[:, frame_width - overlap_width:frame_width] = blended_region.astype(np.uint8)
    
    # 오른쪽 부분은 warped_frame2
    stitched_frame[:, frame_width:] = warped_frame2[:, frame_width:]
    
    # (5) 일반 스티칭 결과 저장
    if want_normal_stitched and out_stitched is not None:
        out_stitched.write(stitched_frame)
    
    # (6) 색상 오버레이 영상
    if want_overlay_stitched and out_overlay is not None:
        # 복사본을 만들어서 오버레이 적용
        overlay_frame = stitched_frame.copy()
        overlay_frame = apply_color_overlay(overlay_frame, frame_width, overlap_width)
        out_overlay.write(overlay_frame)

cap1.release()
cap2.release()

if out_stitched is not None:
    out_stitched.release()
if out_overlay is not None:
    out_overlay.release()

# ================================
# 14. 오버레이 영상 크롭 (want_crop_overlay)
# ================================
if want_overlay_stitched and want_crop_overlay:
    print("\nCalculating optimal crop region for overlay video...")
    cap_overlay = cv2.VideoCapture(overlay_video_path)
    success, frame = cap_overlay.read()
    if not success:
        print("Error: Could not read overlay video for cropping.")
        sys.exit(1)
    
    intersection_mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_mask = (gray > 0).astype(np.uint8)
        intersection_mask = cv2.bitwise_and(intersection_mask, current_mask)
        success, frame = cap_overlay.read()
    
    cap_overlay.release()
    
    contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    else:
        print("No valid stitching region found for overlay crop.")
        sys.exit(1)
    
    print(f"Overlay crop region: x={x}, y={y}, w={w}, h={h}")
    
    # 크롭 비디오 저장
    cap_overlay = cv2.VideoCapture(overlay_video_path)
    out_cropped_overlay = cv2.VideoWriter(cropped_overlay_video_path, fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap_overlay.read()
        if not ret:
            break
        cropped_frame = frame[y:y+h, x:x+w]
        out_cropped_overlay.write(cropped_frame)
    
    cap_overlay.release()
    out_cropped_overlay.release()
    print(f"Cropped overlay video saved: {cropped_overlay_video_path}")

print("\n=== All done! ===")
