import cv2
import numpy as np
import os
import glob

# 프레임이 저장된 폴더
FRAME_DIR = "frames"
# Homography 행렬 파일
HOMOGRAPHY_FILE = "homography_matrices.npz"
# 결과 이미지 저장 폴더
STITCHED_DIR = "stitched_frames"

# 📌 카메라별 폴더 지정
LEFT_CAM = "left"
CENTER_CAM = "center"

# Homography 행렬 불러오기
homography_data = np.load(HOMOGRAPHY_FILE)
H1 = homography_data["H1"]  # 왼쪽 → 중앙 Homography

# 📌 폴더를 수동으로 매핑하여 가져오기
video_folders = {
    "left": os.path.join(FRAME_DIR, LEFT_CAM),
    "center": os.path.join(FRAME_DIR, CENTER_CAM),
}

# 존재 여부 확인
if not all(os.path.exists(folder) for folder in video_folders.values()):
    print("❌ 지정된 카메라 폴더를 찾을 수 없습니다. 폴더명을 확인하세요.")
else:
    frames_left = sorted(glob.glob(os.path.join(video_folders["left"], "*.jpg")))  # 카메라 1 (왼쪽)
    frames_center = sorted(glob.glob(os.path.join(video_folders["center"], "*.jpg")))  # 카메라 2 (중앙)

    os.makedirs(STITCHED_DIR, exist_ok=True)

    if frames_left and frames_center:
        for i in range(min(len(frames_left), len(frames_center))):
            img_left = cv2.imread(frames_left[i])  # 왼쪽 프레임
            img_center = cv2.imread(frames_center[i])  # 중앙 프레임

            h, w = img_center.shape[:2]  # 중앙 프레임 크기 기준

            # 📌 왼쪽 프레임을 중앙 기준으로 변환
            img_left_warped = cv2.warpPerspective(img_left, H1, (w * 2, h))

            # 📌 중앙 프레임을 변환된 왼쪽 프레임에 오버레이
            stitched_img = img_left_warped.copy()
            stitched_img[0:h, w:w*2] = img_center  # 중앙 프레임을 원래 위치에 배치

            # 결과 저장
            frame_name = f"stitched_{i:04d}.jpg"
            output_path = os.path.join(STITCHED_DIR, frame_name)
            cv2.imwrite(output_path, stitched_img)

        print(f"✅ 왼쪽-중앙 프레임 스티칭 완료 → {STITCHED_DIR}")

    else:
        print("❌ 일부 프레임을 찾을 수 없습니다.")
