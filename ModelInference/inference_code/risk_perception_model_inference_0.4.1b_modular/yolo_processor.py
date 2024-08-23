import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from my_utils import plot_one_box
from tracker import ObjectTracker

class YOLOProcessor:
    def __init__(self, model_path, track_classes, conf_threshold=0.25, tracker=None):
        self.model = torch.hub.load('/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/yolov5', 'custom', path=model_path, source='local')
        self.track_classes = track_classes
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image at {image_path} could not be read.")
            return

        img_height, img_width = img.shape[:2]
        results = self.model(img)

        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > self.conf_threshold:
                class_name = self.model.names[int(cls)]
                if class_name in self.track_classes:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox_xywh, conf.item(), class_name))

        # DeepSort를 통한 객체 추적
        tracks = self.deepsort.update_tracks(detections, frame=img)

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

            # 바운딩 박스 비율 계산
            box_area = (x2 - x1) * (y2 - y1)
            image_area = img_width * img_height
            area_ratio = box_area / image_area

            # 객체 인식 메시지 출력
            det_conf = track.get_det_conf()
            if det_conf is not None:
                print(f"Detected: {class_name} with confidence {det_conf:.2f} at {image_path},"
                      f" {class_name} ratio in image: {area_ratio:.2f}")
            else:
                print(f"Detected: {class_name} with unknown confidence at {image_path}")

            # 객체 상태 추적
            self.tracker.track_object(class_name, area_ratio)

        # 결과 저장
        output_path = image_path.replace('input', 'output')
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to {output_path}")
