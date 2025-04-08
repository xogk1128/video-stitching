import cv2
import numpy as np
import pickle

# 카메라 파라미터 교정 후 세 동영상 스티칭
# -------------------------------
# 경로 및 파일명 설정
# -------------------------------
path = 'videos/'
video_left = path + 'left_0314.mp4'
video_center = path + 'center_0314.mp4'
video_right = path + 'right_0314.mp4'
output_video_name = 'panorama_output_test.mp4'

# -------------------------------
# 각 동영상별 캘리브레이션 결과 로드
# -------------------------------
calib_file_left = 'calibration_left.pkl'
calib_file_center = 'calibration_left.pkl'
calib_file_right = 'calibration_right.pkl'

with open(calib_file_left, 'rb') as f:
    cameraMatrix_left, distCoeffs_left = pickle.load(f)
with open(calib_file_center, 'rb') as f:
    cameraMatrix_center, distCoeffs_center = pickle.load(f)
with open(calib_file_right, 'rb') as f:
    cameraMatrix_right, distCoeffs_right = pickle.load(f)

# -------------------------------
# 동영상 캡처 객체 생성
# -------------------------------
cap_left = cv2.VideoCapture(video_left)
cap_center = cv2.VideoCapture(video_center)
cap_right = cv2.VideoCapture(video_right)

if not (cap_left.isOpened() and cap_center.isOpened() and cap_right.isOpened()):
    print("동영상 파일 중 하나 이상을 열 수 없습니다.")
    exit()

# -------------------------------
# 첫 프레임 읽기 및 왜곡 보정 (호모그래피 계산용)
# -------------------------------
ret_left, frame_left = cap_left.read()
ret_center, frame_center = cap_center.read()
ret_right, frame_right = cap_right.read()

if not (ret_left and ret_center and ret_right):
    print("프레임 읽기 오류.")
    exit()

frame_left = cv2.undistort(frame_left, cameraMatrix_left, distCoeffs_left)
frame_center = cv2.undistort(frame_center, cameraMatrix_center, distCoeffs_center)
frame_right = cv2.undistort(frame_right, cameraMatrix_right, distCoeffs_right)

# -------------------------------
# 특징점 검출 및 매칭: SIFT + knnMatch + Lowe's ratio test
# -------------------------------
sift = cv2.SIFT_create()
kp_left, des_left = sift.detectAndCompute(frame_left, None)
kp_center, des_center = sift.detectAndCompute(frame_center, None)
kp_right, des_right = sift.detectAndCompute(frame_right, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches_left = bf.knnMatch(des_left, des_center, k=2)
matches_right = bf.knnMatch(des_right, des_center, k=2)

ratio_thresh = 0.75
good_matches_left = [m for m, n in matches_left if m.distance < ratio_thresh * n.distance]
good_matches_right = [m for m, n in matches_right if m.distance < ratio_thresh * n.distance]

def get_matched_points(matches, kp_src, kp_dst):
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts

src_pts_left, dst_pts_left = get_matched_points(good_matches_left, kp_left, kp_center)
src_pts_right, dst_pts_right = get_matched_points(good_matches_right, kp_right, kp_center)

H_left, _ = cv2.findHomography(src_pts_left, dst_pts_left, cv2.RANSAC, 5.0)
H_right, _ = cv2.findHomography(src_pts_right, dst_pts_right, cv2.RANSAC, 5.0)

# -------------------------------
# 파노라마 캔버스 크기 결정
# -------------------------------
h, w = frame_center.shape[:2]

def get_corners(frame, H):
    h_frame, w_frame = frame.shape[:2]
    corners = np.float32([[0, 0], [w_frame, 0], [w_frame, h_frame], [0, h_frame]]).reshape(-1, 1, 2)
    if H is not None:
        transformed = cv2.perspectiveTransform(corners, H)
    else:
        transformed = corners
    return transformed

corners_left = get_corners(frame_left, H_left)
corners_center = get_corners(frame_center, None)
corners_right = get_corners(frame_right, H_right)

all_corners = np.concatenate((corners_left, corners_center, corners_right), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0, 1]], dtype=np.float32)

panorama_width = x_max - x_min
panorama_height = y_max - y_min

# --- 접합부 블렌딩에 사용할 overlap (픽셀)
overlap = 150

# 중앙 영상은 warp_center
# 그러므로 중앙 영상의 좌측 상단 좌표는 다음과 같이 결정됨.
center_x = int(-x_min)
center_y = int(-y_min)
center_w = frame_center.shape[1]
center_h = frame_center.shape[0]

# -------------------------------
# 출력 동영상 설정
# -------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap_center.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (panorama_width, panorama_height))

# 동영상의 시작 프레임으로 이동
cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap_center.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0

while True:
    ret_center, frame_center = cap_center.read()
    if not ret_center:
        break

    cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not (ret_left and ret_right):
        break

    # 각 동영상 왜곡 보정 - Camera Calibration(calibration/02_calculate..)
    frame_left = cv2.undistort(frame_left, cameraMatrix_left, distCoeffs_left)
    frame_center = cv2.undistort(frame_center, cameraMatrix_center, distCoeffs_center)
    frame_right = cv2.undistort(frame_right, cameraMatrix_right, distCoeffs_right)
    
    # 각 영상의 파노라마 좌표계로 투영 (translation 포함)
    warp_left = cv2.warpPerspective(frame_left, translation @ H_left, (panorama_width, panorama_height))
    warp_center = cv2.warpPerspective(frame_center, translation, (panorama_width, panorama_height))
    warp_right = cv2.warpPerspective(frame_right, translation @ H_right, (panorama_width, panorama_height))
    
    # -------------------------------
    # 접합부 영역의 인덱스를 재조정하여 블렌딩 수행
    # -------------------------------
    # 전체 파노라마를 초기화 (검은색)
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    
    # 1. 왼쪽 단독 영역: [0, center_x)
    if center_x > 0:
        panorama[:, :center_x] = warp_left[:, :center_x]
    
    # 2. 왼쪽–중앙 접합부: [center_x, center_x+overlap]
    #    - 왼쪽 영상의 해당 영역: warp_left[:, center_x:center_x+overlap]
    #    - 중앙 영상의 해당 영역: warp_center[:, center_x:center_x+overlap]
    if center_x + overlap <= panorama_width:
        alpha = np.linspace(1, 0, overlap, endpoint=True)
        alpha = alpha[np.newaxis, :, np.newaxis]  # shape (1, overlap, 1)
        left_part = warp_left[:, center_x:center_x+overlap].astype(np.float32)
        center_left = warp_center[:, center_x:center_x+overlap].astype(np.float32)
        blended = left_part * alpha + center_left * (1 - alpha)
        panorama[:, center_x:center_x+overlap] = blended.astype(np.uint8)
    
    # 3. 중앙 단독 영역: [center_x+overlap, center_x+center_w-overlap)
    center_main_start = center_x + overlap
    center_main_end = center_x + center_w - overlap
    if center_main_end > center_main_start:
        panorama[:, center_main_start:center_main_end] = warp_center[:, center_main_start:center_main_end]
    
    # 4. 중앙–오른쪽 접합부: [center_x+center_w-overlap, center_x+center_w]
    if center_x + center_w <= panorama_width:
        alpha = np.linspace(1, 0, overlap, endpoint=True)
        alpha = alpha[np.newaxis, :, np.newaxis]
        center_right = warp_center[:, center_x+center_w-overlap:center_x+center_w].astype(np.float32)
        right_part = warp_right[:, center_x+center_w-overlap:center_x+center_w].astype(np.float32)
        blended = center_right * alpha + right_part * (1 - alpha)
        panorama[:, center_x+center_w-overlap:center_x+center_w] = blended.astype(np.uint8)
    
    # 5. 오른쪽 단독 영역: [center_x+center_w, panorama_width)
    if center_x + center_w < panorama_width:
        panorama[:, center_x+center_w:] = warp_right[:, center_x+center_w:]
    
    output_video.write(panorama)
    cv2.imshow("Panorama", panorama)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap_left.release()
cap_center.release()
cap_right.release()
output_video.release()
cv2.destroyAllWindows()
