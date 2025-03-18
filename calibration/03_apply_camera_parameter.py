import pickle
import os
import numpy as np

# 카메라 파라미터 결과값 유효성 검토


def analyze_camera_parameters(file_path, image_width, image_height):
    # 파일 존재 여부 체크
    if not os.path.exists(file_path):
        print("에러: 파일이 존재하지 않습니다:", file_path)
        return
    
    # 파라미터 파일 불러오기 (cameraMatrix, distCoeffs)
    with open(file_path, "rb") as f:
        try:
            cameraMatrix, distCoeffs = pickle.load(f)
        except Exception as e:
            print("파라미터 파일 로딩 중 오류 발생:", e)
            return
    
    print("----- 카메라 파라미터 분석 -----")
    print("Camera Matrix:")
    print(cameraMatrix)
    print("Distortion Coeffs:")
    print(distCoeffs)
    
    # 카메라 행렬에서 초점 거리와 주점 추출
    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]
    
    # 이미지 중심 예상 값 계산
    expected_cx = image_width / 2.0
    expected_cy = image_height / 2.0
    
    print(f"\n입력 이미지 해상도: {image_width} x {image_height}")
    print(f"예상 이미지 중심: ({expected_cx:.2f}, {expected_cy:.2f})")
    print(f"측정된 초점 거리: fx = {fx:.2f}, fy = {fy:.2f}")
    print(f"측정된 주점: cx = {cx:.2f}, cy = {cy:.2f}")
    
    # 기준값 및 체크
    all_good = True
    
    # 1. 초점 거리는 양수여야 함.
    if fx <= 0 or fy <= 0:
        print("경고: 초점 거리는 양수여야 합니다.")
        all_good = False
    else:
        print("초점 거리 값은 양수입니다.")
    
    # 2. 주점이 이미지 중심과 크게 벗어나지 않아야 함.
    # 예를 들어, 이미지 폭의 10% 정도 이내로 차이 나는지 확인.
    tol_x = 0.1 * image_width
    tol_y = 0.1 * image_height
    if abs(cx - expected_cx) > tol_x or abs(cy - expected_cy) > tol_y:
        print("경고: 주점(principal point)이 예상 이미지 중심에서 크게 벗어났습니다.")
        all_good = False
    else:
        print("주점(principal point)이 이미지 중심에 근접합니다.")
    
    # 3. 왜곡 계수는 보통 절대값이 1 미만이어야 함.
    # 왜곡 계수가 배열 형태인 경우 flatten 하여 각 값을 확인.
    dist_flat = distCoeffs.flatten() if isinstance(distCoeffs, np.ndarray) else np.array(distCoeffs).flatten()
    for i, val in enumerate(dist_flat):
        print(f"왜곡 계수 k{i+1}: {val:.4f}")
        if abs(val) > 1:
            print(f"경고: 왜곡 계수 k{i+1}의 절대값이 1보다 큽니다. (비정상적일 수 있습니다.)")
            all_good = False
    
    # 최종 결과 출력
    if all_good:
        print("\n분석 결과: 측정된 카메라 파라미터가 유의미한 값으로 보입니다.")
    else:
        print("\n분석 결과: 일부 파라미터 값이 예상 범위를 벗어났습니다. 캘리브레이션 과정을 재검토해보시기 바랍니다.")

if __name__ == "__main__":
    file_path = input("파라미터 파일의 경로와 파일명을 입력하세요: ").strip()
    
    try:
        image_width = float(input("체커보드 캘리브레이션에 사용된 이미지의 width를 입력하세요: "))
        image_height = float(input("체커보드 캘리브레이션에 사용된 이미지의 height를 입력하세요: "))
    except ValueError:
        print("숫자 값을 입력해야 합니다.")
        exit()
    
    analyze_camera_parameters(file_path, image_width, image_height)
