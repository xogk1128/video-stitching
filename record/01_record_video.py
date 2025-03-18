import cv2
import threading

# RTSP 스트림 URL
# cam1_url = "rtsp://admin:asdf1346@@192.168.10.103/stream1/out.h264"
cam2_url = "rtsp://admin:asdf1346@@192.168.10.104/stream1/out.h264"

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 버퍼 설정 (60초)
buffer_size = 60 * 30  # FPS 30 기준 30초

def record_video(cam_url, output_file, buffer_size):
    cap = cv2.VideoCapture(cam_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:  # FPS를 가져오지 못한 경우 기본값 설정
        fps = 30.0

    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < buffer_size:
        ret, frame = cap.read()
        if not ret:
            break
        video_writer.write(frame)
        frame_count += 1

    cap.release()
    video_writer.release()
    print(f"{output_file} 저장 완료.")

# thread1 = threading.Thread(target=record_video, args=(cam1_url, "output_left.mp4", buffer_size))
thread1 = threading.Thread(target=record_video, args=(cam2_url, "output_right.mp4", buffer_size))

thread1.start()

thread1.join()

print("녹화 완료.")
