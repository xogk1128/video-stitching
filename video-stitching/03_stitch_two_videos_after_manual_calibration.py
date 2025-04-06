import cv2
import numpy as np

# 수동 파라미터를 바탕으로 교정후 스티칭

path = 'videos/'
video1_path = path + 'test_left.mp4'
video2_path = path + 'test_right.mp4'
output_video_path = 'stitched_video_after_manual_callibration.mp4'
cropped_output_video_path = 'stitched_video_manual_calibration_cropped.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap1.get(cv2.CAP_PROP_FPS)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

if not (ret1 and ret2):
    print("Cannot read initial frames.")
    exit()

K = np.array([[frame_width, 0, frame_width / 2],
              [0, frame_width, frame_height / 2],
              [0, 0, 1]])

dist_coeffs1 = np.array([-0.44, -0.06, 0.0, -0.0185, 0.015])
dist_coeffs2 = np.array([-0.48, -0.100, 0.0, 0, 0.01])

# dist_coeffs2 = np.array([-5.47304037e-01,  5.98534003e-01,  4.59542259e-04,  1.05272523e-04, -6.30835742e-01])

def undistort_frame(frame, K, dist_coeffs):
    return cv2.undistort(frame, K, dist_coeffs)

frame1_undistorted = undistort_frame(frame1, K, dist_coeffs1)
frame2_undistorted = undistort_frame(frame2, K, dist_coeffs2)

gray1 = cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray1 = clahe.apply(gray1)
gray2 = clahe.apply(gray2)

sift = cv2.SIFT_create(nfeatures=12000)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
matches = flann.knnMatch(des1, des2, k=2)

good_matches, pts1, pts2 = [], [], []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
        good_matches.append(m)



pts1, pts2 = np.float32(pts1), np.float32(pts2)
F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 2.0, 0.99)
pts1_inliers = pts1[mask_f.ravel() == 1]
pts2_inliers = pts2[mask_f.ravel() == 1]

H, _ = cv2.findHomography(pts2_inliers, pts1_inliers, cv2.RANSAC, 4.0)

# Reset videos
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

out_w, out_h = frame_width * 2, frame_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

def create_feather_mask(width, height, overlap_width):
    mask = np.ones((height, width), dtype=np.float32)
    for i in range(overlap_width):
        alpha = (i / overlap_width)
        mask[:, width - overlap_width + i] = 1 - alpha
    return cv2.merge([mask, mask, mask])

overlap_width = 150  # Adjust as needed

print("Stitching video frames...")

final_stitched_frame = None  # To store last frame for cropping

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not (ret1 and ret2):
        break

    frame1_undistorted = undistort_frame(frame1, K, dist_coeffs1)
    frame2_undistorted = undistort_frame(frame2, K, dist_coeffs2)

    warped_frame2 = cv2.warpPerspective(frame2_undistorted, H, (out_w, out_h))
    stitched_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    stitched_frame[:, :frame_width] = frame1_undistorted

    mask1 = create_feather_mask(frame_width, frame_height, overlap_width)
    mask2 = 1 - mask1

    blended_region = (stitched_frame[:, frame_width - overlap_width:frame_width].astype(np.float32) * mask1[:, -overlap_width:] +
                      warped_frame2[:, frame_width - overlap_width:frame_width].astype(np.float32) * mask2[:, frame_width - overlap_width:frame_width])

    stitched_frame[:, frame_width - overlap_width:frame_width] = blended_region.astype(np.uint8)
    stitched_frame[:, frame_width:] = warped_frame2[:, frame_width:]

    final_mask = (cv2.cvtColor(stitched_frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)*255

    out.write(stitched_frame)
    final_stitched_frame = stitched_frame.copy()  # keep last frame for cropping calculation

cap1.release()
cap2.release()
out.release()
print("Calculating optimal (smallest) cropping region...")

cap = cv2.VideoCapture(output_video_path)
success, frame = cap.read()

# Initialize intersection mask to all ones
intersection_mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

while success:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_mask = (gray > 0).astype(np.uint8)
    intersection_mask = cv2.bitwise_and(intersection_mask, current_mask)
    success, frame = cap.read()

cap.release()

# Find bounding rectangle of intersection (smallest rectangle)
contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
else:
    raise ValueError("No valid stitching region found.")

print(f"Optimal crop region (smallest rectangle): x={x}, y={y}, w={w}, h={h}")

# Crop the original stitched video to the smallest rectangle
cap = cv2.VideoCapture(output_video_path)
cropped_output_video_path = 'stitched_video_cropped.mp4'
out_cropped = cv2.VideoWriter(cropped_output_video_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cropped_frame = frame[y:y+h, x:x+w]
    out_cropped.write(cropped_frame)

cap.release()
out_cropped.release()

print(f"Cropped video saved: {cropped_output_video_path}")
