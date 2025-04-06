import cv2
import numpy as np

# 카메라 보정 없이 동영상 스티칭
# -------------------------------
# 동영상 파일 경로 및 출력 파일명 설정
# -------------------------------
path = 'videos/'
video1_path = path + 'test_left.mp4'
video2_path = path + 'test_right.mp4'

# 사용자 지정 출력 파일 이름
output_video_path = 'panorama_video_blended.mp4'               # 스티칭 결과 영상
cropped_output_video_path = 'panorama_video_blended_cropped.mp4'  # 검은 영역 제거된 영상

# -------------------------------------------------
def compute_homography(img1, img2):
    """
    img1: 기준 이미지 (첫 번째 동영상 프레임)
    img2: 투영할 이미지 (두 번째 동영상 프레임)
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        return H
    else:
        return None

def create_feather_mask(width, height, overlap_width):
    mask = np.ones((height, width), dtype=np.float32)
    for i in range(overlap_width):
        alpha = i / overlap_width
        mask[:, width - overlap_width + i] = 1 - alpha
    return cv2.merge([mask, mask, mask])

# -------------------------------
# 1. 동영상 로딩 및 첫 프레임으로 호모그래피 계산
# -------------------------------
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
if not ret1 or not ret2:
    print("첫 프레임을 읽어올 수 없습니다.")
    exit()

H = compute_homography(frame1, frame2)
if H is None:
    print("호모그래피 계산 실패")
    exit()

height, width, _ = frame1.shape
panorama_width = width + frame2.shape[1]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap1.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (panorama_width, height))

overlap_width = 150

cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("파노라마 동영상(블렌딩 적용) 생성 중...")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    warped_frame2 = cv2.warpPerspective(frame2, H, (panorama_width, height))
    panorama = np.zeros((height, panorama_width, 3), dtype=np.uint8)
    panorama[:, :width] = frame1

    mask1 = create_feather_mask(width, height, overlap_width)
    mask2 = 1 - mask1

    blended_region = (
        panorama[:, width - overlap_width:width].astype(np.float32) * mask1[:, -overlap_width:] +
        warped_frame2[:, width - overlap_width:width].astype(np.float32) * mask2[:, width - overlap_width:width]
    )
    panorama[:, width - overlap_width:width] = blended_region.astype(np.uint8)
    panorama[:, width:] = warped_frame2[:, width:]

    out.write(panorama)

out.release()
cap1.release()
cap2.release()

print(f"파노라마 동영상 저장 완료: {output_video_path}")

# -------------------------------
# 2. 검은 영역 제거 위한 크롭 처리
# -------------------------------
print("검은 영역 제거를 위한 크롭 영역 계산 중...")

cap = cv2.VideoCapture(output_video_path)
success, frame = cap.read()
if not success:
    print("크롭 대상 동영상을 읽어올 수 없습니다.")
    exit()

intersection_mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_mask = (gray > 0).astype(np.uint8)
    intersection_mask = cv2.bitwise_and(intersection_mask, current_mask)
cap.release()

contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    print(f"크롭 영역: x={x}, y={y}, w={w}, h={h}")
else:
    raise ValueError("유효한 스티칭 영역을 찾을 수 없습니다.")

# -------------------------------
# 3. 크롭된 동영상 생성
# -------------------------------
print("크롭 동영상 생성 중...")

cap = cv2.VideoCapture(output_video_path)
cropped_out = cv2.VideoWriter(cropped_output_video_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cropped_frame = frame[y:y+h, x:x+w]
    cropped_out.write(cropped_frame)

cap.release()
cropped_out.release()

print(f"크롭 동영상 저장 완료: {cropped_output_video_path}")
cv2.destroyAllWindows()
