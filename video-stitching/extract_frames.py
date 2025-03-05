import cv2
import os
import glob

VIDEO_DIR = "videos"
FRAME_DIR = "frames"

# 프레임 추출
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 파일을 열 수 없습니다: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS 탐색
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수
    print(f"📹 {video_path}: FPS={fps}, 총 프레임={frame_count}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 읽을 프레임이 없으면 종료
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"✅ 프레임 추출 완료: {video_path} → {output_folder}")

# 모든 동영상 처리
video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
if not video_files:
    print("❌ videos 폴더에 .mp4 파일이 없습니다.")

for video_file in video_files:
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_folder = os.path.join(FRAME_DIR, video_name)
    extract_frames(video_file, output_folder)

print("🎬 모든 동영상의 프레임 추출이 완료되었습니다!")
