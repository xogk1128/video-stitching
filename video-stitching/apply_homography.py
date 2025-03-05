import cv2
import numpy as np
import os
import glob

# 프레임이 저장된 폴더
FRAME_DIR = "frames"
# Homography 행렬 파일
HOMOGRAPHY_FILE = "homography_matrices.npz"
# 변환된 프레임 저장 폴더
WARPED_DIR = "warped_frames"

# 📌 카메라별 폴더 지정
LEFT_CAM = "left"
CENTER_CAM = "center"
RIGHT_CAM = "right"

# Homography 행렬 불러오기
homography_data = np.load(HOMOGRAPHY_FILE)
H1 = homography_data["H1"]  # 왼쪽 → 중앙
H2 = homography_data["H2"]  # 오른쪽 → 중앙

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

    os.makedirs(WARPED_DIR, exist_ok=True)

    if frames_left and frames_center and frames_right:
        for i in range(min(len(frames_left), len(frames_center), len(frames_right))):
            img_left = cv2.imread(frames_left[i])  # 왼쪽 프레임
            img_center = cv2.imread(frames_center[i])  # 중앙 프레임
            img_right = cv2.imread(frames_right[i])  # 오른쪽 프레임

            h, w = img_center.shape[:2]  # 중앙 프레임 크기 기준

            # 📌 왼쪽 프레임을 중앙 기준으로 변환
            img_left_warped = cv2.warpPerspective(img_left, H1, (w * 3, h))
            img_left_warped[:h, w:w * 2] = img_center  # 중앙 프레임을 그대로 유지

            # 📌 오른쪽 프레임을 중앙 기준으로 변환
            img_right_warped = cv2.warpPerspective(img_right, H2, (w * 3, h))
            
            # 📌 최종 파노라마 프레임 만들기 (왼쪽 + 중앙 + 오른쪽)
            panorama_frame = img_left_warped.copy()
            panorama_frame[:h, w * 2:] = img_right_warped[:h, w * 2:]

            # 변환된 프레임 저장
            frame_name = f"frame_{i:04d}.jpg"
            output_path = os.path.join(WARPED_DIR, frame_name)
            cv2.imwrite(output_path, panorama_frame)

        print(f"✅ 모든 프레임이 변환 및 저장 완료 → {WARPED_DIR}")

    else:
        print("❌ 일부 프레임을 찾을 수 없습니다.")
