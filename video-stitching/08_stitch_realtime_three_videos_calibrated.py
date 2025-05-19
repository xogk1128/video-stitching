import cv2
import numpy as np
import pickle
import time

# RTSP 주소 입력
rtsp_right = "rtsp://admin:asdf1346@@192.168.9.101/stream1/out.h264"
rtsp_left = "rtsp://admin:asdf1346@@192.168.9.102/stream1/out.h264"
rtsp_center = "rtsp://admin:asdf1346@@192.168.9.103/stream1/out.h264"

# 캘리브레이션 파일 로드 (세 카메라 모두 같은 파라미터라고 가정)
with open('calibration.pkl', 'rb') as f:
    cameraMatrix, distCoeffs = pickle.load(f)

# RTSP 스트림 열기
cap_left = cv2.VideoCapture(rtsp_left)
cap_center = cv2.VideoCapture(rtsp_center)
cap_right = cv2.VideoCapture(rtsp_right)

if not (cap_left.isOpened() and cap_center.isOpened() and cap_right.isOpened()):
    print("RTSP 스트림을 열 수 없습니다.")
    exit()

# 첫 프레임에서 호모그래피 계산
def get_initial_homographies():
    print("초기 프레임으로 호모그래피 계산 중...")

    ret_left, frame_left = cap_left.read()
    ret_center, frame_center = cap_center.read()
    ret_right, frame_right = cap_right.read()

    if not (ret_left and ret_center and ret_right):
        print("초기 프레임 로딩 실패")
        exit()

    frame_left = cv2.undistort(frame_left, cameraMatrix, distCoeffs)
    frame_center = cv2.undistort(frame_center, cameraMatrix, distCoeffs)
    frame_right = cv2.undistort(frame_right, cameraMatrix, distCoeffs)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def match_pair(img1, img2):
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    H_left = match_pair(frame_left, frame_center)
    H_right = match_pair(frame_right, frame_center)

    return H_left, H_right, frame_center.shape

H_left, H_right, frame_shape = get_initial_homographies()

# 파노라마 크기 계산
h, w = frame_shape[:2]

def get_corners(frame, H):
    h_frame, w_frame = frame.shape[:2]
    corners = np.float32([[0, 0], [w_frame, 0], [w_frame, h_frame], [0, h_frame]]).reshape(-1, 1, 2)
    if H is not None:
        return cv2.perspectiveTransform(corners, H)
    return corners

corners_left = get_corners(np.zeros((h, w, 3)), H_left)
corners_center = get_corners(np.zeros((h, w, 3)), None)
corners_right = get_corners(np.zeros((h, w, 3)), H_right)

all_corners = np.concatenate((corners_left, corners_center, corners_right), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0, 1]], dtype=np.float32)

panorama_width = x_max - x_min
panorama_height = y_max - y_min
overlap = 150
center_x, center_y = int(-x_min), int(-y_min)
center_w = w

# 실시간 루프
print("실시간 파노라마 생성 시작")
while True:
    ret_left, frame_left = cap_left.read()
    ret_center, frame_center = cap_center.read()
    ret_right, frame_right = cap_right.read()

    if not (ret_left and ret_center and ret_right):
        print("스트림 프레임을 읽을 수 없습니다.")
        break

    frame_left = cv2.undistort(frame_left, cameraMatrix, distCoeffs)
    frame_center = cv2.undistort(frame_center, cameraMatrix, distCoeffs)
    frame_right = cv2.undistort(frame_right, cameraMatrix, distCoeffs)

    warp_left = cv2.warpPerspective(frame_left, translation @ H_left, (panorama_width, panorama_height))
    warp_center = cv2.warpPerspective(frame_center, translation, (panorama_width, panorama_height))
    warp_right = cv2.warpPerspective(frame_right, translation @ H_right, (panorama_width, panorama_height))

    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # 1. 왼쪽 영역
    if center_x > 0:
        panorama[:, :center_x] = warp_left[:, :center_x]

    # 2. 왼쪽–중앙 접합
    if center_x + overlap <= panorama_width:
        alpha = np.linspace(1, 0, overlap)[None, :, None]
        blended = warp_left[:, center_x:center_x+overlap].astype(np.float32) * alpha + \
                  warp_center[:, center_x:center_x+overlap].astype(np.float32) * (1 - alpha)
        panorama[:, center_x:center_x+overlap] = blended.astype(np.uint8)

    # 3. 중앙 영역
    main_start = center_x + overlap
    main_end = center_x + center_w - overlap
    if main_end > main_start:
        panorama[:, main_start:main_end] = warp_center[:, main_start:main_end]

    # 4. 중앙–오른쪽 접합
    if center_x + center_w <= panorama_width:
        alpha = np.linspace(1, 0, overlap)[None, :, None]
        blended = warp_center[:, center_x+center_w-overlap:center_x+center_w].astype(np.float32) * alpha + \
                  warp_right[:, center_x+center_w-overlap:center_x+center_w].astype(np.float32) * (1 - alpha)
        panorama[:, center_x+center_w-overlap:center_x+center_w] = blended.astype(np.uint8)

    # 5. 오른쪽 단독 영역
    if center_x + center_w < panorama_width:
        panorama[:, center_x+center_w:] = warp_right[:, center_x+center_w:]

    cv2.imshow("Live Panorama", panorama)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_center.release()
cap_right.release()
cv2.destroyAllWindows()
