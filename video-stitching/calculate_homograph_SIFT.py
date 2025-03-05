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
    sift = cv2.SIFT_create()

    # 특징점 및 디스크립터 추출
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # BFMatcher를 사용한 특징점 매칭
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 매칭된 특징점 정렬 (거리순)
    matches = sorted(matches, key=lambda x: x.distance)

    # 최상위 매칭점 50개 선택
    good_matches = matches[:50]

    # 매칭된 좌표 추출
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Homography 행렬 계산 (RANSAC 사용)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
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
