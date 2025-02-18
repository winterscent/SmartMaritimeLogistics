"""
import torch
import cv2
import numpy as np
import time
from PIL import Image

# YOLOv5 모델 로드
model = torch.load('best.pt')
model.eval()


# 영상 처리 및 저장
def process_video(input_path, output_path, model, conf_threshold=0.25):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    def plot_one_box(x, img, color=(128, 128, 128), label=None, line_thickness=3):
        # 바운딩 박스를 그리는 함수
        tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

        # 모델 추론
        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        # 추론 시간 계산
        inference_time = end_time - start_time
        print(f"Inference Time: {inference_time:.3f} seconds")

        # 바운딩 박스 그리기
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > conf_threshold:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)

        # 프레임 저장
        out.write(frame)

    cap.release()
    out.release()


# 영상 파일 경로 설정
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

# 영상 처리
process_video(input_video_path, output_video_path, model)


# 처리된 영상 재생 (선택 사항)
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


play_video(output_video_path)
"""
