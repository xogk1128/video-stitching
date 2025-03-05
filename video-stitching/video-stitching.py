import cv2
import numpy as np
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{frame_idx:04d}.jpg", frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {video_path}")

def stitch_images(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        height, width, _ = img2.shape
        result = cv2.warpPerspective(img1, H, (width * 2, height))
        result[0:height, 0:width] = img2
        
        return result
    else:
        return None

def process_frames(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    frame_list1 = sorted(os.listdir(folder1))
    frame_list2 = sorted(os.listdir(folder2))
    
    for idx, (frame1_name, frame2_name) in enumerate(zip(frame_list1, frame_list2)):
        img1 = cv2.imread(os.path.join(folder1, frame1_name))
        img2 = cv2.imread(os.path.join(folder2, frame2_name))
        
        stitched_frame = stitch_images(img1, img2)
        if stitched_frame is not None:
            cv2.imwrite(f"{output_folder}/stitched_{idx:04d}.jpg", stitched_frame)
        else:
            print(f"Skipping frame {idx} due to stitching failure")

def crop_images(input_folder, output_folder, crop_x, crop_y, crop_w, crop_h):
    os.makedirs(output_folder, exist_ok=True)
    
    for img_name in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        cropped_img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        cv2.imwrite(os.path.join(output_folder, img_name), cropped_img)

def frames_to_video(frame_folder, output_video_path, fps):
    frame_list = sorted(os.listdir(frame_folder))
    first_frame = cv2.imread(os.path.join(frame_folder, frame_list[0]))
    
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_name in frame_list:
        frame = cv2.imread(os.path.join(frame_folder, frame_name))
        out.write(frame)
    
    out.release()
    print(f"Video saved as {output_video_path}")

# 실행 흐름
extract_frames("videos/1.mp4", "frames_cam1")
extract_frames("videos/2.mp4", "frames_cam2")
process_frames("frames_cam1", "frames_cam2", "stitched_frames")
crop_images("stitched_frames", "cropped_frames", 200, 100, 500, 500)
frames_to_video("cropped_frames", "final_panorama_video.avi", fps=30)