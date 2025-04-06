import cv2
import numpy as np
import pickle

# 카메라 파라미터로 교정 후 동영상 스티칭

# 캘리브레이션 파일에서 보정값 불러오기
with open("calibration_left.pkl", "rb") as f:
    cameraMatrix_left, distCoeffs_left = pickle.load(f)

with open("calibration_right.pkl", "rb") as f:
    cameraMatrix_right, distCoeffs_right = pickle.load(f)

# 동영상 파일 경로 설정
path = 'videos/'
video1_path = path + 'test_left.mp4'
video2_path = path + 'test_right.mp4'
output_video_path = 'stitched_video_robust.mp4'
cropped_output_video_path = 'stitched_video_cropped.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap1.get(cv2.CAP_PROP_FPS)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

if not (ret1 and ret2):
    print("Cannot read initial frames.")
    exit()

def undistort_frame(frame, cameraMatrix, distCoeffs):
    return cv2.undistort(frame, cameraMatrix, distCoeffs)

# 보정값을 사용해 초기 프레임 왜곡 보정
frame1_undistorted = undistort_frame(frame1, cameraMatrix_left, distCoeffs_left)
frame2_undistorted = undistort_frame(frame2, cameraMatrix_right, distCoeffs_right)

# 특징점 검출을 위한 전처리: 그레이스케일 변환 및 CLAHE 적용
gray1 = cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray1 = clahe.apply(gray1)
gray2 = clahe.apply(gray2)

# SIFT 특징점 검출 및 서술자 계산
sift = cv2.SIFT_create(nfeatures=12000)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN 기반 매칭
flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
matches = flann.knnMatch(des1, des2, k=2)

good_matches, pts1, pts2 = [], [], []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
        good_matches.append(m)

pts1, pts2 = np.float32(pts1), np.float32(pts2)
# RANSAC을 사용해 Fundamental Matrix 계산 및 inlier 검출
F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 2.0, 0.99)
pts1_inliers = pts1[mask_f.ravel() == 1]
pts2_inliers = pts2[mask_f.ravel() == 1]

# inlier에 대해 호모그래피 계산 (오른쪽 프레임을 왼쪽 프레임에 맞춤)
H, _ = cv2.findHomography(pts2_inliers, pts1_inliers, cv2.RANSAC, 4.0)

# 동영상의 시작 프레임으로 되돌림
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

out_w, out_h = frame_width * 2, frame_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

def create_feather_mask(width, height, overlap_width):
    mask = np.ones((height, width), dtype=np.float32)
    for i in range(overlap_width):
        alpha = i / overlap_width
        mask[:, width - overlap_width + i] = 1 - alpha
    return cv2.merge([mask, mask, mask])

overlap_width = 50  # 필요에 따라 조정

print("Stitching video frames...")

final_stitched_frame = None  # 크롭 계산을 위한 마지막 프레임 저장

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        break

    # 각 프레임에서 보정값 적용
    frame1_undistorted = undistort_frame(frame1, cameraMatrix_left, distCoeffs_left)
    frame2_undistorted = undistort_frame(frame2, cameraMatrix_right, distCoeffs_right)

    # 오른쪽 프레임을 호모그래피 H로 워핑
    warped_frame2 = cv2.warpPerspective(frame2_undistorted, H, (out_w, out_h))
    
    # 왼쪽 프레임과 워핑된 오른쪽 프레임을 블렌딩
    stitched_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    stitched_frame[:, :frame_width] = frame1_undistorted

    mask1 = create_feather_mask(frame_width, frame_height, overlap_width)
    mask2 = 1 - mask1

    blended_region = (
        stitched_frame[:, frame_width - overlap_width:frame_width].astype(np.float32) * mask1[:, -overlap_width:] +
        warped_frame2[:, frame_width - overlap_width:frame_width].astype(np.float32) * mask2[:, frame_width - overlap_width:frame_width]
    )
    stitched_frame[:, frame_width - overlap_width:frame_width] = blended_region.astype(np.uint8)
    stitched_frame[:, frame_width:] = warped_frame2[:, frame_width:]

    out.write(stitched_frame)
    final_stitched_frame = stitched_frame.copy()  # 마지막 프레임 저장

cap1.release()
cap2.release()
out.release()

print("Calculating optimal (smallest) cropping region...")

# 전체 영상에서 유효 영역(픽셀 값이 있는 영역)을 찾아 크롭 영역 계산
cap = cv2.VideoCapture(output_video_path)
success, frame = cap.read()

intersection_mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
while success:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_mask = (gray > 0).astype(np.uint8)
    intersection_mask = cv2.bitwise_and(intersection_mask, current_mask)
    success, frame = cap.read()
cap.release()

# 유효 영역의 최소 사각형 영역 찾기
contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
else:
    raise ValueError("No valid stitching region found.")

print(f"Optimal crop region (smallest rectangle): x={x}, y={y}, w={w}, h={h}")

# 원본 스티칭 영상을 계산된 영역으로 크롭하여 저장
cap = cv2.VideoCapture(output_video_path)
cropped_output_video_path = 'stitched_video_cropped.mp4'
out_cropped = cv2.VideoWriter(cropped_output_video_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cropped_frame = frame[y:y+h, x:x+w]
    out_cropped.write(cropped_frame)

cap.release()
out_cropped.release()

print(f"Cropped video saved: {cropped_output_video_path}")
