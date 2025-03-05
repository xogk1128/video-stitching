import cv2
import numpy as np
import os
import glob
# 프레임이 저장된 폴더
FRAME_DIR = "frames"
# Homography 행렬 저장 파일
HOMOGRAPHY_FILE = "homography_matrices.npz"
LEFT_CAM = "left"     # 왼쪽 카메라
CENTER_CAM = "center"   # 중앙 카메라
RIGHT_CAM = "right"    # 오른쪽 카메라

# 특징점 추출 및 매칭 함수
def find_homography(image1, image2):
    orb = cv2.ORB_create(nfeatures=5000)  # ORB 특징점 검출기
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # FLANN 기반 매칭
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test 적용
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:  # 매칭 수가 너무 적으면 실패
        print("❌ 충분한 특징점 매칭이 없습니다.")
        return None, None, None, None

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC Threshold 5.0 → 3.0으로 조정
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    return H, good_matches, keypoints1, keypoints2



video_folders = { 
    "left": os.path.join(FRAME_DIR, LEFT_CAM),
    "center": os.path.join(FRAME_DIR, CENTER_CAM),
    "right": os.path.join(FRAME_DIR, RIGHT_CAM)
}

# 존재 여부 예외 처리
if not all(os.path.exists(folder) for folder in video_folders.values()):
    print("❌ 지정된 카메라 폴더를 찾을 수 없습니다. 폴더명을 확인하세요.")
else:
    frames_left = sorted(glob.glob(os.path.join(video_folders["left"], "*.jpg")))  # 카메라 1 (왼쪽)
    frames_center = sorted(glob.glob(os.path.join(video_folders["center"], "*.jpg")))  # 카메라 2 (중앙)
    frames_right = sorted(glob.glob(os.path.join(video_folders["right"], "*.jpg")))  # 카메라 3 (오른쪽)

    if frames_left and frames_center and frames_right:
        img_left = cv2.imread(frames_left[0])  # 왼쪽 첫 프레임
        img_center = cv2.imread(frames_center[0])  # 중앙 첫 프레임
        img_right = cv2.imread(frames_right[0])  # 오른쪽 첫 프레임

        # 왼쪽 → 중앙 Homography 계산
        H1, matches1, kpts1, kpts2 = find_homography(img_left, img_center)

        # 오른쪽 → 중앙 Homography 계산
        H2, matches2, kpts3, kpts4 = find_homography(img_right, img_center)

        # Homography 행렬 저장
        np.savez(HOMOGRAPHY_FILE, H1=H1, H2=H2)
        print(f"✅ Homography 행렬이 '{HOMOGRAPHY_FILE}' 파일로 저장되었습니다.")
    else:
        print("❌ 프레임을 찾을 수 없습니다.")
