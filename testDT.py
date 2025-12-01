from indy_utils import indydcp_client as client
import json
from time import sleep
import threading
import numpy as np
import cv2

# Keras Model for O/X Classification ----------------
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

MODEL_PATH = "keras_Model.h5"
LABEL_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)
class_names = open(LABEL_PATH, "r").readlines()


#############################################
# 1) Robot Utility
#############################################

def IsMoveDone(indy_t):
    while True:
        status = indy_t.get_robot_status()
        sleep(0.5)
        if status['movedone'] == True:
            break

def grip(hold, indy_t):
    indy_t.set_do(2, hold)


#############################################
# 2) Box ROI 자동 검출 함수
#############################################

def detect_box(image):
    orig = image.copy()
    h, w = image.shape[:2]

    cut_top = int(h * 0.20)
    work = image[cut_top:h, 0:w]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    kernel = np.ones((7,7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best_box = None

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        if area < 5000:
            continue

        aspect = w_box / float(h_box)
        if 1.2 < aspect < 3.5:   # 상자 비율 필터
            if area > best_area:
                best_area = area
                best_box = (x, y, w_box, h_box)

    if best_box is None:
        return None, None

    x, y, w_box, h_box = best_box
    y = y + cut_top

    # 내부 여백 제거
    mx = int(w_box * 0.05)
    my = int(h_box * 0.08)

    x2 = max(0, x + mx)
    y2 = max(0, y + my)
    w2 = w_box - mx*2
    h2 = h_box - my*2

    roi = orig[y2:y2+h2, x2:x2+w2]

    return roi, (x2, y2, w2, h2)


#############################################
# 3) ROI 내부 원형 물체 검출 + 분류
#############################################

def detect_and_classify(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 2)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=60,
        param1=80, param2=30,
        minRadius=20, maxRadius=60
    )

    results = []
    if circles is None:
        return []

    circles = np.uint16(np.around(circles))

    for (cx, cy, r) in circles[0, :]:
        x1, y1 = max(0, cx-r), max(0, cy-r)
        x2, y2 = min(roi.shape[1], cx+r), min(roi.shape[0], cy+r)
        crop = roi[y1:y2, x1:x2]

        pil = Image.fromarray(crop).convert("RGB")
        pil = ImageOps.fit(pil, (224,224), Image.Resampling.LANCZOS)

        data = np.asarray(pil).astype(np.float32)
        data = (data / 127.5) - 1
        data = np.expand_dims(data, axis=0)

        pred = model.predict(data)
        idx = np.argmax(pred)
        label = class_names[idx].strip()
        conf = float(pred[0][idx])

        results.append({
            "cx": cx,
            "cy": cy,
            "r": r,
            "label": label,
            "conf": conf
        })

    return results


#############################################
# 4) Pixel → Meter 변환
#############################################

BOX_W = 0.085   # 8.5cm
BOX_H = 0.150   # 15cm

def pixel_to_meter(cx, cy, box_w_px, box_h_px):
    px = cx * (BOX_W / box_w_px)
    py = (box_h_px - cy) * (BOX_H / box_h_px)
    return px, py


#############################################
# 5) Cam → Robot 좌표 변환
#############################################

def conCamtoRobo(px, py):
    zero_x = 0.50790
    zero_y = -0.02140
    zero_z = 0.500

    robot_x = zero_x + px
    robot_y = zero_y + py
    robot_z = zero_z

    return [robot_x, robot_y, robot_z, -180, 0, 180]


#############################################
# 6) Pick 동작
#############################################

def picknPlace(indy_t, target_pose):
    indy_t.connect()
    indy_t.go_home()
    IsMoveDone(indy_t)

    # 접근 자세
    app = target_pose.copy()
    app[2] += 0.10   # 위쪽 10cm 접근

    indy_t.task_move_to(app)
    IsMoveDone(indy_t)

    indy_t.task_move_to(target_pose)
    IsMoveDone(indy_t)

    grip(True, indy_t)

    indy_t.task_move_to(app)
    IsMoveDone(indy_t)

    indy_t.disconnect()


#############################################
# 7) 전체 파이프라인 (MAIN)
#############################################

def main_function(indy_t):

    # ---- 0) 카메라에서 이미지 1장 캡처 (예시) ----
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("카메라 캡처 실패")
        return

    # ---- 1) 상자 자동 검출 ----
    roi, box_rect = detect_box(frame)
    if roi is None:
        print("상자 검출 실패")
        return

    x_box, y_box, w_box, h_box = box_rect

    # ---- 2) ROI 안에서 원형 물체 + 분류 ----
    detections = detect_and_classify(roi)
    if len(detections) == 0:
        print("물체 검출 실패")
        return

    # 일단 첫 번째 물체만 pick
    d = detections[0]

    print(f"Detected {d['label']} at px=({d['cx']},{d['cy']}) conf={d['conf']}")

    # ---- 3) 픽셀 → 미터 변환 ----
    px_m, py_m = pixel_to_meter(d['cx'], d['cy'], w_box, h_box)

    # ---- 4) 카메라 → 로봇 변환 ----
    robot_pose = conCamtoRobo(px_m, py_m)

    print("Robot pose:", robot_pose)

    # ---- 5) 로봇 Pick ----
    picknPlace(indy_t, robot_pose)


#############################################
# Program Start
#############################################

robot_ip = "192.168.3.6"
robot_name = "NRMK-Indy7"

indy1 = client.IndyDCPClient(robot_ip, robot_name)
main_function(indy1)
