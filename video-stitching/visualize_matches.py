import cv2
import numpy as np
import glob
import os

# 프레임이 저장된 폴더
FRAME_DIR = "frames"
# 저장된 Homography 행렬 파일
HOMOGRAPHY_FILE = "homography_matrices.npz"

# 📌 카메라별 폴더 지정
LEFT_CAM = "left"
CENTER_CAM = "center"
RIGHT_CAM = "right"

# 📌 특징점 매칭 시각화 함수
def draw_matches(img1, img2, keypoints1, keypoints2, matches, title):
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 📌 폴더를 수동으로 매핑하여 가져오기
video_folders = {
    "left": os.path.join(FRAME_DIR, LEFT_CAM),
    "center": os.path.join(FRAME_DIR, CENTER_CAM),
    "right": os.path.join(FRAME_DIR, RIGHT_CAM)
}

# 존재 여부 확인
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

        # SIFT 생성
        sift = cv2.SIFT_create()

        # 특징점 및 디스크립터 추출
        keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
        keypoints_center, descriptors_center = sift.detectAndCompute(img_center, None)
        keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

        # BFMatcher를 사용한 특징점 매칭
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # 📌 왼쪽 → 중앙 특징점 매칭
        matches_left_center = bf.match(descriptors_left, descriptors_center)
        matches_left_center = sorted(matches_left_center, key=lambda x: x.distance)[:50]

        # 📌 오른쪽 → 중앙 특징점 매칭
        matches_right_center = bf.match(descriptors_right, descriptors_center)
        matches_right_center = sorted(matches_right_center, key=lambda x: x.distance)[:50]

        # 📌 시각화 실행
        draw_matches(img_left, img_center, keypoints_left, keypoints_center, matches_left_center, "Left ↔ Center")
        draw_matches(img_right, img_center, keypoints_right, keypoints_center, matches_right_center, "Right ↔ Center")

    else:
        print("❌ 프레임을 찾을 수 없습니다.")
