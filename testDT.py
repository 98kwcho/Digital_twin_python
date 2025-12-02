from indy_utils import indydcp_client as client
import json
from time import sleep
import threading
import numpy as np
import cv2

# =======================
# Keras Classification
# =======================
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

MODEL_PATH = "keras_Model.h5"
LABEL_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)
class_names = open(LABEL_PATH, "r").readlines()


#############################################
# 0) 원본 이미지 전체를 분류 (keras 원본 코드 유지)
#############################################

def classify_whole_image(frame):

    pil_img = Image.fromarray(frame).convert("RGB")

    size = (224, 224)
    pil_img = ImageOps.fit(pil_img, size, Image.Resampling.LANCZOS)

    img_arr = np.asarray(pil_img)
    norm = (img_arr.astype(np.float32) / 127.5) - 1

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = norm

    pred = model.predict(data)
    idx = np.argmax(pred)
    label = class_names[idx].strip()
    conf = float(pred[0][idx])

    return label, conf


#############################################
# 1) Robot Utility
#############################################

def IsMoveDone(indy_t):
    while True:
        status = indy_t.get_robot_status()
        sleep(0.5)
        if status["movedone"] == True:
            break

def grip(hold, indy_t):
    indy_t.set_do(2, hold)


#############################################
# 2) Box ROI 자동 검출
#############################################

def detect_box(image):

    orig = image.copy()
    h, w = image.shape[:2]

    cut_top = int(h * 0.20)
    work = image[cut_top:h]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    kernel = np.ones((7,7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 2)

    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best_box = None

    for cnt in contours:
        x,y,w_box,h_box = cv2.boundingRect(cnt)
        area = w_box*h_box
        if area < 5000:
            continue

        aspect = w_box / float(h_box)
        if 1.2 < aspect < 3.5:
            if area > best_area:
                best_area = area
                best_box = (x,y,w_box,h_box)

    if best_box is None:
        return None, None

    x,y,w_box,h_box = best_box
    y = y + cut_top

    mx = int(w_box * 0.05)
    my = int(h_box * 0.08)

    x2 = x + mx
    y2 = y + my
    w2 = w_box - mx * 2
    h2 = h_box - my * 2

    roi = orig[y2:y2+h2, x2:x2+w2]

    return roi, (x2, y2, w2, h2)


#############################################
# 3) ROI 내부 원형 물체 탐지 (분류 없음)
#############################################

def detect_circles(roi):

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

    for (cx, cy, r) in circles[0]:
        results.append({
            "cx": int(cx),
            "cy": int(cy),
            "r": int(r)
        })

    return results


#############################################
# 4) Pixel → Meter 변환
#############################################

BOX_W = 0.085
BOX_H = 0.150

def pixel_to_meter(cx, cy, box_w_px, box_h_px):
    px = cx * (BOX_W / box_w_px)
    py = (box_h_px - cy) * (BOX_H / box_h_px)
    return px, py


#############################################
# 5) Cam → Robot 좌표 변환
#############################################

def conCamtoRobo(px, py):
    base_x = 0.50790
    base_y = -0.02140
    base_z = 0.500

    return [
        base_x + px,
        base_y + py,
        base_z,
        -180, 0, 180
    ]


#############################################
# 6) Pick 동작
#############################################

def picknPlace(indy_t, pose):
    indy_t.connect()
    indy_t.go_home()
    IsMoveDone(indy_t)

    app = pose.copy()
    app[2] += 0.10

    indy_t.task_move_to(app)
    IsMoveDone(indy_t)

    indy_t.task_move_to(pose)
    IsMoveDone(indy_t)

    grip(True, indy_t)

    indy_t.task_move_to(app)
    IsMoveDone(indy_t)

    indy_t.disconnect()


#############################################
# 7) 전체 파이프라인 MAIN
#############################################

def main_function(indy_t):

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Camera Fail")
        return

    # ----------------------------
    # Step 1) 원본 분류
    # ----------------------------
    label, conf = classify_whole_image(frame)
    print(f"[CLASSIFY] → {label} , conf={conf:.3f}")

    # O인지 X인지에 따라 pick 여부 결정 가능

    # ----------------------------
    # Step 2) 상자 자동 검출
    # ----------------------------
    roi, box_rect = detect_box(frame)
    if roi is None:
        print("상자 검출 실패")
        return
    
    x_box, y_box, w_box, h_box = box_rect

    # ----------------------------
    # Step 3) 원형 탐지 (좌표용)
    # ----------------------------
    circles = detect_circles(roi)
    if len(circles) == 0:
        print("원형 물체 없음")
        return

    d = circles[0]
    print(f"[CIRCLE] cx={d['cx']} cy={d['cy']} r={d['r']}")

    # ----------------------------
    # Step 4) 픽셀 → 미터 변환
    # ----------------------------
    px_m, py_m = pixel_to_meter(d["cx"], d["cy"], w_box, h_box)

    # ----------------------------
    # Step 5) 카메라 → 로봇 좌표
    # ----------------------------
    robot_pose = conCamtoRobo(px_m, py_m)
    print("Target Robot Pose:", robot_pose)

    # ----------------------------
    # Step 6) pick 동작
    # ----------------------------
    picknPlace(indy_t, robot_pose)



#############################################
# 실행 시작
#############################################

robot_ip = "192.168.3.6"
robot_name = "NRMK-Indy7"

indy1 = client.IndyDCPClient(robot_ip, robot_name)
main_function(indy1)
