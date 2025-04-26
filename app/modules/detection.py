import numpy as np
import cv2
from PIL import Image

def yolo_detect(image, model, conf_thresh=0.7):
    results = model.predict(source=image, conf=conf_thresh, save=False, imgsz=640, iou=0.3)
    result_img = results[0].plot()
    masks = results[0].masks.xy if results[0].masks else []
    return Image.fromarray(result_img[..., ::-1]), masks

def order_corners(pts):
    pts = np.array(pts).reshape(-1, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def binarize(image):
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def crop_objects(image, segments):
    crops = []
    binarized_crops = []
    img_np = np.array(image)

    for i, seg in enumerate(segments):
        if len(seg) < 4:
            continue
        try:
            box = order_corners(seg)
            width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
            height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
            dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
            M = cv2.getPerspectiveTransform(box, dst_pts)
            warped = cv2.warpPerspective(img_np, M, (width, height))
            pil_crop = Image.fromarray(warped)
            binarized_crop = binarize(pil_crop)
            crops.append((f"object_{i+1}.jpg", pil_crop))
            binarized_crops.append((f"object_{i+1}.jpg", binarized_crop))
        except Exception as e:
            print(f"Polygon {i} failed: {e}")
            continue
    return crops, binarized_crops