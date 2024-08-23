import torch
import cv2

# YOLOv5 모델 로드
model = torch.hub.load('../yolov5', 'custom', path='model/best_0.3.0.pt', source='local')


def detect_objects(image_path, model):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image at {image_path} could not be read.")
        return

    # 모델 추론
    results = model(img)

    # 결과 처리
    if len(results.xyxy[0]) > 0:
        print(f"Objects detected in {image_path}:")
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            print(f" - {label}")
            # 바운딩 박스 그리기
            plot_one_box(xyxy, img, label=label)

        # 결과 이미지 저장
        output_path = image_path.replace('input', 'output')
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to {output_path}")
    else:
        print(f"No objects detected in {image_path}")


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


# 이미지 경로 설정 및 감지 실행
image_path = 'input/004.png'  # 여기에 사용할 이미지 경로를 입력하세요
detect_objects(image_path, model)
