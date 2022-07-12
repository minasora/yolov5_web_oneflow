import oneflow as torch
from yolo import YOLOv5
from PIL import Image
import numpy as np
import yolo
import cv2
from flowvision import transforms

classes = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")


def draw_detection(image_dir, boxes):
    draw_img = cv2.imread(image_dir)
    for i in range(len(boxes['label'])):
        bbox = boxes['box'][i]
        label = boxes['label'][i]
        score = boxes['score'][i]
        bbox_color = (255, 0, 255)
        label_color = (255, 0, 255)
        cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=2)
        labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        if bbox[1] - labelSize[1] - 3 < 0:
            cv2.rectangle(draw_img,
                          (bbox[0], bbox[1] + 2),
                          (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                          color=label_color,
                          thickness=-1
                          )
            cv2.putText(draw_img, label,
                        (bbox[0], bbox[1] + labelSize + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=-1
                        )
        else:
            cv2.rectangle(draw_img,
                          (bbox[0], bbox[1] - labelSize[1] - 3),
                          (bbox[0] + labelSize[0], bbox[1] - 3),
                          color=label_color,
                          thickness=-1
                          )
            cv2.putText(draw_img, label,
                        (bbox[0], bbox[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=1
                        )
    cv2.imwrite('static/test.jpg', draw_img)
def infer(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load('model').to(device)
    model.eval()
    img = Image.open(img)
    img = transforms.ToTensor()(img).to(device)
    image = [img]
    results = model(image)
    results = results[0][0]
    results['scores'] = np.array(results['scores'].cpu().numpy())
    boxes = {'box': [], 'label': [], 'score': []}
    for i in range(len(results['scores'])):
        if results['scores'][i] > 0.5:
            boxes['box'].append(results['boxes'][i, :].int().numpy().tolist())
            boxes['label'].append(classes[results['labels'][i].item()])
            boxes['score'].append(results['scores'][i])
    draw_detection('static/test.jpg',boxes)


