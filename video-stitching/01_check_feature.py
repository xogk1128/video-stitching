import cv2
import numpy as np
import pickle
import os

# 특이점 검출 확인

# 영상 폴더
video_path = "videos/"

# 영상 경로
left_video_path = video_path + "1.mp4"
right_video_path = video_path + "2.mp4"

# 카메라 파라미터 파일 경로
left_calib_file = "calibration_1.pkl"
right_calib_file = "calibration_2.pkl"

def main():
    # 첫번째 프레임 읽기
    cap1 = cv2.VideoCapture(left_video_path)
    cap2 = cv2.VideoCapture(right_video_path)

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        print("Error: Could not read initial frames from the videos.")
        return

    cap1.release()
    cap2.release()

    # 카메라 파라미터 불러오기
    if not os.path.exists(left_calib_file):
        print(f"Error: {left_calib_file} not found.")
        return
    if not os.path.exists(right_calib_file):
        print(f"Error: {right_calib_file} not found.")
        return

    with open(left_calib_file, "rb") as f:
        left_data = pickle.load(f)  # (cameraMatrix, distCoeffs) 형태라 가정
        K_left, dist_left = left_data[0], left_data[1]

    with open(right_calib_file, "rb") as f:
        right_data = pickle.load(f)
        K_right, dist_right = right_data[0], right_data[1]

    print("Left Camera Matrix:\n", K_left)
    print("Left Distortion Coeffs:\n", dist_left)
    print("Right Camera Matrix:\n", K_right)
    print("Right Distortion Coeffs:\n", dist_right)

    # 첫 프레임 왜곡 보정
    frame1_undistorted = cv2.undistort(frame1, K_left, dist_left)
    frame2_undistorted = cv2.undistort(frame2, K_right, dist_right)

    # 그레이스케일 및 대비 보정(선택)
    gray1 = cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2GRAY)

    # CLAHE로 국소 대비 향상 (선택 사항)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray1 = clahe.apply(gray1)
    gray2 = clahe.apply(gray2)

    # SIFT 특징점 검출 및 디스크립터 추출
    sift = cv2.SIFT_create(nfeatures=12000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN 매칭 (KNN)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(des1, des2, k=2)

    # 좋은 매칭 선별(Lowe's ratio test)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Total matches: {len(knn_matches)} / Good matches after ratio test: {len(good_matches)}")

    # 매칭 시각화 (drawMatches)
    # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS → 단일 점 제외, 매칭선 위주
    matched_image = cv2.drawMatches(
        frame1_undistorted, kp1,
        frame2_undistorted, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 결과 표시
    cv2.imshow("Matched Keypoints (Press any key to exit)", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 결과 이미지 저장
    cv2.imwrite("matched_keypoints.png", matched_image)
    print("Saved 'matched_keypoints.png' with visualized matches.")

if __name__ == "__main__":
    main()
