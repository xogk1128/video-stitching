import numpy as np
import cv2 as cv
import glob
import pickle
import os

# 교정용 카메라 파라미터 계산
# 저장된 카메라 파라미터는 calibration.pkl 로 저장됨

# -----------------------------
# 1. 체스보드 파라미터 설정
# -----------------------------
# 내부 코너 (가로 x 세로)
chessboardSize = (8, 6)

# 실제 체스보드 한 칸의 크기 (mm)
size_of_chessboard_squares_mm = 30

# 캘리브레이션에 사용할 이미지 경로 설정 (png, jpg 등 확장자에 맞게 변경 가능)
image_path_pattern = '../images/*.png'

# 사용 중인 이미지(프레임) 해상도 (예: 1920 x 1080)
frameSize = (1920, 1080)

# -----------------------------
# 2. 체스보드 3D 좌표 생성
# -----------------------------
# 예: (0,0,0), (1,0,0), (2,0,0) ... (7,5,0)
#   -> Z축은 0(평면), X, Y축은 체스보드 칸 크기만큼 곱해줌
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp = objp * size_of_chessboard_squares_mm

# -----------------------------
# 3. 코너 검출 결과 저장용 리스트
# -----------------------------
objpoints = []  # 3D 공간상의 좌표
imgpoints = []  # 2D 이미지 좌표

# -----------------------------
# 4. 체스보드 코너 검출
# -----------------------------
# 지정한 폴더에서 이미지 목록 불러오기
image_files = glob.glob(image_path_pattern)

if not image_files:
    print(f"Error: '{image_path_pattern}' 패턴으로 불러올 이미지가 없습니다.")
    exit()

# OpenCV 코너 서브픽셀 보정 파라미터
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in image_files:
    # 이미지 읽기
    img = cv.imread(fname)
    if img is None:
        print(f"이미지 로드 실패: {fname}")
        continue

    # 그레이스케일 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        # 코너 감지 성공 시, 서브픽셀 정확도로 코너 좌표 보정
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 3D 좌표 및 2D 좌표 리스트에 추가
        objpoints.append(objp)
        imgpoints.append(corners_refined)

        # 디버깅용: 이미지에 코너를 그려서 확인
        cv.drawChessboardCorners(img, chessboardSize, corners_refined, ret)
        cv.imshow('Checkerboard Detection', img)
        cv.waitKey(500)  # 0.5초 대기
    else:
        print(f"[실패] 체스보드 코너를 찾지 못했습니다: {fname}")

cv.destroyAllWindows()

# -----------------------------
# 5. 카메라 캘리브레이션
# -----------------------------
# 코너가 하나도 감지되지 않았다면 종료
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("체스보드 코너 감지에 성공한 이미지가 없으므로 캘리브레이션을 진행할 수 없습니다.")
    exit()

# 카메라 보정
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints,    # 3D 점
    imgpoints,    # 2D 점
    frameSize,    # 이미지 크기
    None, None    # 초기 카메라 행렬, 왜곡 계수
)

if not ret:
    print("카메라 캘리브레이션에 실패했습니다.")
    exit()

print("카메라 캘리브레이션 성공!")
print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coeffs:\n", distCoeffs)

# -----------------------------
# 6. 결과 저장
# -----------------------------
# pickle로 행렬과 왜곡 계수를 파일에 저장
save_dir = "calibration_result"
os.makedirs(save_dir, exist_ok=True)

pickle.dump((cameraMatrix, distCoeffs), open(os.path.join(save_dir, "calibration.pkl"), "wb"))
pickle.dump(cameraMatrix, open(os.path.join(save_dir, "cameraMatrix.pkl"), "wb"))
pickle.dump(distCoeffs, open(os.path.join(save_dir, "distCoeffs.pkl"), "wb"))

print(f"보정 결과가 '{save_dir}' 폴더에 저장되었습니다.")