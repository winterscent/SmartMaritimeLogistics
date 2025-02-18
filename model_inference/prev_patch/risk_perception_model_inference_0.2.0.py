import os
import time
import torch
import cv2

# YOLOv5 모델 로드
model = torch.hub.load('yolov5', 'custom', path='best_0.2.1.pt', source='local')


def process_image(image_path, model, conf_threshold=0.25):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image at {image_path} could not be read.")
        return

    # 모델 추론
    results = model(img)

    # 결과 처리 및 바운딩 박스 그리기
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > conf_threshold:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=(255, 0, 0), line_thickness=2)
    # 결과 저장 혹은 출력
    output_path = image_path.replace('input', 'output')
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")


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
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main():
    input_directory = 'input'
    processed_files = set()

    while True:
        if os.path.exists(input_directory):
            for filename in os.listdir(input_directory):
                if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in processed_files:
                    image_path = os.path.join(input_directory, filename)
                    process_image(image_path, model)
                    processed_files.add(filename)
        else:
            print(f"Directory {input_directory} not found. Retrying...")

        # 주기적으로 디렉토리 확인 (예: 10초마다)
        time.sleep(10)


if __name__ == "__main__":
    main()
