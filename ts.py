import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

IMAGE_PATH = "ng_w-samples/30.jpg"
MODEL_PATH = "keras_Model.h5"
LABEL_PATH = "labels.txt"

print("[INFO] loading model...")
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABEL_PATH, "r").readlines()


#############################################
# 0) 원본 이미지 분류 (가장 먼저 수행)
#############################################
def classify_full_image(img):

    pil = Image.fromarray(img).convert("RGB")
    pil = ImageOps.fit(pil, (224,224), Image.Resampling.LANCZOS)

    arr = np.asarray(pil).astype(np.float32)
    norm = (arr / 127.5) - 1

    data = np.ndarray((1,224,224,3), dtype=np.float32)
    data[0] = norm

    pred = model.predict(data)
    index = np.argmax(pred)
    label = class_names[index].strip()
    conf = float(pred[0][index])

    return label, conf


#############################################
# 1) 상자 자동 검출
#############################################
def detect_box(image):
    orig = image.copy()
    h, w = image.shape[:2]

    cut_top = int(h * 0.20)
    work = image[cut_top:h, :]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 35, 5
    )

    kernel = np.ones((7,7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best_box = None

    for cnt in contours:
        x,y,w_box,h_box = cv2.boundingRect(cnt)
        area=w_box*h_box

        if area < 5000:
            continue

        aspect = w_box / float(h_box)
        if 1.2 < aspect < 3.5:
            if area > best_area:
                best_area = area
                best_box = (x,y,w_box,h_box)

    if best_box is None:
        raise RuntimeError("상자를 찾지 못했습니다.")

    x,y,w_box,h_box = best_box
    y = y + cut_top

    # 내부 여백 제거
    mx = int(w_box * 0.05)
    my = int(h_box * 0.08)

    x2 = x + mx
    y2 = y + my
    w2 = w_box - mx*2
    h2 = h_box - my*2

    roi = orig[y2:y2+h2, x2:x2+w2]

    return roi, (x2,y2,w2,h2), th, orig


#############################################
# 2) 원형 탐지만 수행 (분류 없음)
#############################################
def detect_circles(roi):

    result = roi.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),2)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=60,
        param1=80, param2=30,
        minRadius=20, maxRadius=60
    )

    detections = []
    if circles is None:
        return result, []

    circles = np.uint16(np.around(circles))

    for (cx,cy,r) in circles[0]:
        detections.append((cx,cy,r))
        cv2.circle(result,(cx,cy),r,(0,255,0),2)
        cv2.circle(result,(cx,cy),2,(0,255,255),3)

    return result, detections


#############################################
# MAIN
#############################################
def main():

    img = cv2.imread(IMAGE_PATH)

    # --- Step 1: 분류 (가장 먼저!)
    label, conf = classify_full_image(img)
    print(f"[CLASSIFY] → {label}, conf={conf:.3f}")

    # --- Step 2: 상자 검출
    roi, box_rect, th, dbg = detect_box(img)
    x,y,w,h = box_rect
    cv2.rectangle(dbg,(x,y),(x+w,y+h),(0,255,0),3)

    # --- Step 3: ROI에서 원형 탐지 (좌표만)
    circ_img, circles = detect_circles(roi)

    print("[CIRCLES] detected:", circles)

    cv2.imshow("threshold", th)
    cv2.imshow("box_debug", dbg)
    cv2.imshow("ROI", roi)
    cv2.imshow("circles", circ_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
