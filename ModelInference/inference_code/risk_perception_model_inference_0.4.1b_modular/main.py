import os
import time
from yolo_processor import YOLOProcessor
from tracker import ObjectTracker

def main():
    input_directory = '/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/input/imagedata/fl1'
    processed_files = set()

    track_classes = ['car', 'truck', 'van', 'forklift', 'fire', 'smoke']
    model_path = '/Users/winterscent/DevWorkSpace/PythonWorkSpace/SmartMaritimeLogistics/ModelInference/model/best_0.3.1.pt'

    # ObjectTracker 인스턴스 생성
    tracker = ObjectTracker(frame_check_threshold=3, fire_smoke_frame_check_threshold=5,
                            alert_threshold=0.01, warning_ratio=0.03, danger_ratio=0.05)

    # YOLOProcessor 인스턴스 생성
    yolo_processor = YOLOProcessor(model_path=model_path, track_classes=track_classes, tracker=tracker)

    while True:
        if os.path.exists(input_directory):
            files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
            files.sort(key=lambda x: os.path.getmtime(x))

            for file_path in files:
                filename = os.path.basename(file_path)
                if filename not in processed_files:
                    yolo_processor.process_image(file_path)
                    processed_files.add(filename)
        else:
            print(f"Directory {input_directory} not found. Retrying...")

        time.sleep(10)

if __name__ == "__main__":
    main()
