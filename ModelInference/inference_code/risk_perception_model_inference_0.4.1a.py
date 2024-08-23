import os
import time
import torch
import cv2
import numpy as np
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv5 모델 로드
model = torch.hub.load('/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/yolov5',
                       'custom',
                       path='/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/model/best_0.3.1.pt',
                       source='local')

# DeepSort 초기화
deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100)

# 추적 대상 클래스
track_classes = ['car', 'truck', 'van', 'forklift', 'fire', 'smoke']
fire_smoke_frame_check_threshold = 15  # 화재/연기 감지 프레임

# 경고 상태 관리
tracked_objects = {}
object_paths = {}


def process_image(image_path, model, conf_threshold=0.25):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image at {image_path} could not be read.")
        return

    img_height, img_width = img.shape[:2]

    # 모델 추론
    results = model(img)

    # 바운딩 박스 및 클래스 정보 추출
    detections = []
    class_names = []
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > conf_threshold:
            class_name = model.names[int(cls)]
            if class_name in track_classes:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox_xywh, conf.item(), class_name))
                class_names.append(class_name)

    # DeepSort를 통한 객체 추적
    tracks = deepsort.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltwh()
        class_name = track.get_det_class()
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        label = f'{class_name} {track_id}'
        plot_one_box([x1, y1, x2, y2], img, label=label, color=(255, 0, 0), line_thickness=2)

        # 경로 기록
        if track_id not in object_paths:
            object_paths[track_id] = deque(maxlen=30)
        object_paths[track_id].append((x1, y1, x2, y2))

        # 이동 방향 계산
        if len(object_paths[track_id]) > 1:
            (prev_x1, prev_y1, prev_x2, prev_y2) = object_paths[track_id][-2]
            (curr_x1, curr_y1, curr_x2, curr_y2) = object_paths[track_id][-1]

            prev_center = ((prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2)
            curr_center = ((curr_x1 + curr_x2) / 2, (curr_y1 + curr_y2) / 2)

            prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
            curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)

            center_distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
            area_change = curr_area - prev_area

            # 중심이 한가운데로 이동하고 넓이가 커지면 가까워진다고 판단
            if center_distance < 10 and area_change > 0:  # 임계값은 필요에 따라 조정
                print(f"Object {track_id} ({class_name}) is moving closer. "
                      f"Distance: {center_distance:.2f}, Area change: {area_change:.2f}")

        # 객체 인식 메시지 출력
        det_conf = track.get_det_conf()
        if det_conf is not None:
            print(f"Detected: {class_name} with confidence {det_conf:.2f} at {image_path}")
        else:
            print(f"Detected: {class_name} with unknown confidence at {image_path}")

        # Fire와 Smoke 클래스는 지정 프레임 횟수가 지나도 감지되면 경고
        if class_name in ['fire', 'smoke']:
            if track_id not in tracked_objects:
                tracked_objects[track_id] = {'frames_since_first_detection': 0}
            tracked_objects[track_id]['frames_since_first_detection'] += 1
            if tracked_objects[track_id]['frames_since_first_detection'] > fire_smoke_frame_check_threshold:
                print(f"[경고] {class_name} detected for more than {fire_smoke_frame_check_threshold} frames")

    # 결과 저장
    output_path = image_path.replace('input', 'output')
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")


def plot_one_box(x, img, color=(128, 128, 128), label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main():
    input_directory = '/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/input/imagedata/fl3'
    processed_files = set()

    while True:
        if os.path.exists(input_directory):
            files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
            files.sort(key=lambda x: os.path.getmtime(x))

            for file_path in files:
                filename = os.path.basename(file_path)
                if filename not in processed_files:
                    process_image(file_path, model)
                    processed_files.add(filename)
        else:
            print(f"Directory {input_directory} not found. Retrying...")

        time.sleep(10)


if __name__ == "__main__":
    main()
