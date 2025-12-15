from indy_utils import indydcp_client as client
import json
from time import sleep
import threading
import numpy as np
import cv2

# Melsec PLC
import pymcprotocol as mc

# Keras Classification
from keras.models import load_model
from PIL import Image, ImageOps

MODEL_PATH = "keras_Model.h5"
LABEL_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)
class_names = open(LABEL_PATH, "r").readlines()

FIXED_BOX = {
    "x": 120,
    "y": 160,
    "w": 400,
    "h": 250
}

# 0) 원본 이미지 전체를 분류 (keras 이용)
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

# 1) Robot Utility
def IsMoveDone(indy_t):
    while True:
        status = indy_t.get_robot_status()
        sleep(0.5)
        if status["movedone"] == True:
            break

def grip(hold, indy_t):
    indy_t.set_do(2, hold)

# 2) Box ROI 자동 검출
def detect_box(image):
    orig = image.copy()
    x = FIXED_BOX["x"]
    y = FIXED_BOX["y"]
    w = FIXED_BOX["w"]
    h = FIXED_BOX["h"]

    # 이미지 경계 확인
    img_h, img_w = orig.shape[:2]

    if (x + w > img_w) or (y + h > img_h):
        print(f"경고: 박스가 이미지 범위를 벗어납니다. 이미지 크기: {img_w}x{img_h}, 박스: x={x}, y={y}, w={w}, h={h}")
        # 안전한 경계로 조정
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

    roi = orig[y:y+h, x:x+w]

    return roi, (x, y, w, h)

# 3) ROI 내부 원형 물체 탐지
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

# 4) Pixel → Meter 변환
BOX_W = 0.150
BOX_H = 0.085
def pixel_to_meter(cx, cy, box_w_px, box_h_px):
    px = cx * (BOX_W / box_w_px)
    py = (box_h_px - cy) * (BOX_H / box_h_px)
    return px, py

# 5) Cam → Robot 좌표 변환
def conCamtoRobo(px, py):
    base_x = 0.52446
    base_y = 0.27
    base_z = 0.340

    return [
        base_x + py - 0.0130,
        base_y + px + 0.0082,
        base_z,
        -180, 0, 180
    ]

# 6) Pick 동작
def picknPlace(indy_t, pose, label):

    # 불량품인 경우
    if label == "1 ng_w" or label == "3 ng_b":
        app = pose.copy()
        tar = pose.copy()
        ret = pose.copy()
        tar[2] -= 0.07
        ret[2] += 0.02
        app2 = [0.18632, 0.37008, 0.4, -180, 0, 180]
        tar2 = app2.copy()
        tar2[2] -= 0.15


        indy_t.connect()

        indy_t.go_home()
        IsMoveDone(indy_t)

        indy_t.task_move_to(app)
        IsMoveDone(indy_t)

        indy_t.task_move_to(tar)
        IsMoveDone(indy_t)

        grip(True, indy_t)

        indy_t.task_move_to(ret)
        IsMoveDone(indy_t)

        indy_t.task_move_to(app2)
        IsMoveDone(indy_t)

        indy_t.task_move_to(tar2)
        IsMoveDone(indy_t)

        grip(False, indy_t)

        indy_t.task_move_to(app2)
        IsMoveDone(indy_t)

        indy_t.go_home()
        IsMoveDone(indy_t)
        indy_t.disconnect()

    else :
        print("라벨 감지 실패")
# 7) PLC bit메모리 통신
def plc_bitread (plc_t, str):
    bit_vals = plc_t.batchread_bitunits(str, 1)
    return bit_vals[0]

def plc_bitwrite (plc_t, str, result):
    plc_t.batchwrite_bitunits(str, [result])
    return

# 8) 전체 파이프라인 MAIN
def main_function(indy_t, plc_t):
    cap = cv2.VideoCapture(0)
    # TODO 아래 대략적 과정을 PLC 에서 들어오는 신호(카메라 스톱퍼, 로봇 스톱퍼)에 따라 동작을 수행하게 한다.
    while True:
        #----- 카메라 스톱퍼 실린더 신호 들어올 시 -----
        if plc_bitread(plc_t, "B101") == 1:
            ret, frame = cap.read()


            if not ret:
                print("Camera Fail")
                return

            # Step 1) 원본 분류 O인지 X인지에 따라 pick 여부 결정 가능
            label, conf = classify_whole_image(frame)
            print(f"[CLASSIFY] → {label} , conf={conf:.3f}")

            # Step 2) 상자 자동 검출
            roi, box_rect = detect_box(frame)
            if roi is None:
                print("상자 검출 실패")
                return

            x_box, y_box, w_box, h_box = box_rect

            # Step 3) 원형 탐지 (좌표용)
            circles = detect_circles(roi)
            if len(circles) == 0:
                print("원형 물체 없음")
                return

            d = circles[0]
            print(f"[CIRCLE] cx={d['cx']} cy={d['cy']} r={d['r']}")

            # Step 4) 픽셀 → 미터 변환
            px_m, py_m = pixel_to_meter(d["cx"], d["cy"], w_box, h_box)

            # Step 5) 카메라 → 로봇 좌표
            robot_pose = conCamtoRobo(px_m, py_m)
            print("Target Robot Pose:", robot_pose)
            cap.release()
            if label == "1 ng_w" or label == "3 ng_b":
                while plc_bitread(plc_t, "B110") == 0 :
                    plc_bitwrite (plc_t, "B110", 1)
                break

            else :
                while plc_bitread(plc_t, "B112") == 0 :
                    plc_bitwrite (plc_t, "B112", 1)
                break
        # ----- 카메라 스톱퍼 실린더 하강 후 컨베이어 벨트 동작 -----

        # ----- 로봇 스톱퍼 신호 들어올 시 -----
    while True:
        if plc_bitread(plc_t, "B103") == 1:
            print("로봇 동작")
            # Step 6) pick 동작
            picknPlace(indy_t, robot_pose, label)

            while plc_bitread(plc_t, "B111") == 0 :
                plc_bitwrite (plc_t, "B111", 1)
                sleep(0.5)

            print("로봇 동작 over")
            break

        elif  plc_bitread(plc_t, "B106") == 1:
            break
        #break
        # ----- 로봇 스톱퍼 신호 해제 -----

# 실행 시작

robot_ip = "192.168.3.6" # robot IP는 변경될 수 있음
robot_name = "NRMK-Indy7"
indy1 = client.IndyDCPClient(robot_ip, robot_name)

plc = mc.Type3E()
plc.setaccessopt(commtype="binary")
plc.connect("192.168.3.130", 1025)      # PLC IP와 포트는 변경될 수 있음
print("연결 성공")

main_function(indy1, plc)

plc.close()