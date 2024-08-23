import cv2
import torch
from my_utils import plot_one_box
from tracker import ObjectTracker  # track_object가 아니라 ObjectTracker 클래스를 가져옵니다.

class YOLOProcessor:
    def __init__(self, model_path, track_classes, conf_threshold=0.25, tracker=None):
        self.model = torch.hub.load('/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/yolov5', 'custom', path=model_path, source='local')
        self.track_classes = track_classes
        self.conf_threshold = conf_threshold
        self.tracker = tracker

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image at {image_path} could not be read.")
            return

        img_height, img_width = img.shape[:2]
        results = self.model(img)

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > self.conf_threshold:
                class_name = self.model.names[int(cls)]
                if class_name in self.track_classes:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    box_area = (x2 - x1) * (y2 - y1)
                    image_area = img_width * img_height
                    area_ratio = box_area / image_area

                    label = f'{class_name} {conf:.2f}'
                    plot_one_box((x1, y1, x2, y2), img, label=label, color=(255, 0, 0), line_thickness=2)

                    print(f"Detected: {class_name} with confidence {conf:.2f} at {image_path},"
                          f" {class_name} ratio in image: {area_ratio:.2f}")

                    if area_ratio >= self.tracker.alert_threshold:
                        self.tracker.track_object(class_name, area_ratio)

        output_path = image_path.replace('input', 'output')
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to {output_path}")
