import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 이미지 폴더 및 파일 설정
IMG_NAME = 'boat'  # 사용할 이미지 폴더명
IMG_DIR = os.path.join('imgs', IMG_NAME)
RESULT_PATH = os.path.join(IMG_DIR, 'result.jpg')
RESULT_CROP_PATH = os.path.join(IMG_DIR, 'result_crop.jpg')

# 이미지 파일 리스트 불러오기
img_list = []
for ext in ('0*.gif', '0*.png', '0*.jpg'):
    img_list.extend(glob.glob(os.path.join(IMG_DIR, ext)))

img_list = sorted(img_list)
print("Loaded images:", img_list)

# 이미지 로드
imgs = []
plt.figure(figsize=(20, 20))
for i, img_path in enumerate(img_list):
    img = cv2.imread(img_path)
    imgs.append(img)
    plt.subplot(len(img_list) // 3 + 1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 스티칭 모드 설정
mode = cv2.Stitcher_PANORAMA

# OpenCV 버전에 따른 Stitcher 생성
if int(cv2.__version__[0]) == 3:
    stitcher = cv2.createStitcher()
else:
    stitcher = cv2.Stitcher_create()

# 이미지 스티칭
status, stitched = stitcher.stitch(imgs)
if status == 0:
    cv2.imwrite(RESULT_PATH, stitched)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Image")
    plt.show()
else:
    print("Stitching failed. Error code:", status)
    exit()

# 그레이스케일 변환 및 이진화
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
thresh = cv2.medianBlur(thresh, 5)

plt.figure(figsize=(20, 20))
plt.imshow(thresh, cmap='gray')
plt.title("Threshold Image")
plt.show()

# 이미지 크롭
stitched_copy = stitched.copy()
thresh_copy = thresh.copy()
while np.sum(thresh_copy) > 0:
    thresh_copy = thresh_copy[1:-1, 1:-1]
    stitched_copy = stitched_copy[1:-1, 1:-1]

cv2.imwrite(RESULT_CROP_PATH, stitched_copy)

plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(stitched_copy, cv2.COLOR_BGR2RGB))
plt.title("Cropped Stitched Image")
plt.show()

print(f"Saved stitched image: {RESULT_PATH}")
print(f"Saved cropped image: {RESULT_CROP_PATH}")
