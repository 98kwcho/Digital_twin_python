from indy_utils import indydcp_client as client
import json
from time import sleep
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

# 캘리브레이션을 위한 상수들
FIXED_BOX = {
    "x": 120,
    "y": 160,
    "w": 400,
    "h": 250
}

# 실제 박스의 물리적 크기 (미터 단위)
BOX_REAL_WIDTH = 0.150   # 실제 너비 (미터)
BOX_REAL_HEIGHT = 0.085  # 실제 높이 (미터)

# 카메라-로봇 좌표 변환 파라미터 (offset이 좌측 상단 기준)
CAM_TO_ROBOT = {
    "offset_x": 0.52446,    # 좌측 상단 X 좌표
    "offset_y": 0.27,       # 좌측 상단 Y 좌표
    "offset_z": 0.340,      # Z축 높이
    "scale_x": 1.0,         # X축 스케일 조정
    "scale_y": 1.0,         # Y축 스케일 조정
    "flip_x": False,        # X축 반전 여부
    "flip_y": False,         # Y축 반전 여부 (카메라 Y축은 아래방향, 로봇 Y축은 위방향)
    "rotation": 0.0         # 회전 각도 (도)
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

    pred = model.predict(data, verbose=0)
    idx = np.argmax(pred)
    label = class_names[idx].strip()
    conf = float(pred[0][idx])

    return label, conf

# 1) Robot Utility
def IsMoveDone(indy_t):
    while True:
        status = indy_t.get_robot_status()
        sleep(0.1)
        if status["movedone"] == True:
            break

def grip(hold, indy_t):
    indy_t.set_do(2, hold)

# 2) 고정된 Box ROI 추출
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

    # ROI가 비어있는지 확인
    if roi.size == 0:
        return None, None

    return roi, (x, y, w, h)

# 3) ROI 내부 원형 물체 탐지
def detect_circles(roi):
    if roi is None or roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # HoughCircles 파라미터 조정
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=15,
        maxRadius=80
    )

    results = []
    if circles is None:
        return []

    circles = np.uint16(np.around(circles))

    # ROI 내에서 가장 큰 원 선택
    for (cx, cy, r) in circles[0]:
        results.append({
            "cx": int(cx),
            "cy": int(cy),
            "r": int(r),
            "area": np.pi * r * r
        })

    # 면적 기준으로 정렬 (가장 큰 원이 먼저)
    results.sort(key=lambda x: x["area"], reverse=True)

    return results

# 4) 픽셀 좌표 → 물리적 좌표 변환 (미터 단위)
def pixel_to_physical(cx_pixel, cy_pixel, box_width_pixel, box_height_pixel):
    """
    ROI 내 픽셀 좌표를 물리적 좌표로 변환
    offset이 좌측 상단 기준이므로, 픽셀 좌표를 그대로 비율로 변환
    """
    # 픽셀 좌표를 박스 크기에 비례하여 물리적 좌표로 변환
    physical_x = (cx_pixel / box_width_pixel) * BOX_REAL_WIDTH
    physical_y = (cy_pixel / box_height_pixel) * BOX_REAL_HEIGHT

    return physical_x, physical_y

# 5) 물리적 좌표 → 로봇 좌표 변환 (offset이 좌측 상단 기준)
def physical_to_robot(physical_x, physical_y):
    """
    physical_x, physical_y: ROI 좌측 상단(0,0)으로부터의 물리적 거리 (m)
    결과: 로봇 베이스 기준 Task 좌표
    """

    # 1. 기본 오프셋(박스 좌측 상단의 로봇 좌표) 적용
    # 이 시점에서 physical_x, y는 박스 안에서의 실제 이동 거리입니다.

    # X축 계산: 로봇 X축 방향이 카메라 X축과 같다면 더함
    if not CAM_TO_ROBOT["flip_x"]:
        robot_x = CAM_TO_ROBOT["offset_x"] + (physical_x * CAM_TO_ROBOT["scale_x"])
    else:
        robot_x = CAM_TO_ROBOT["offset_x"] - (physical_x * CAM_TO_ROBOT["scale_x"])

    # Y축 계산: 카메라 Y는 아래(+), 로봇 Y는 위(+)인 경우가 많으므로 flip_y=True가 기본
    if not CAM_TO_ROBOT["flip_y"]:
        robot_y = CAM_TO_ROBOT["offset_y"] + (physical_y * CAM_TO_ROBOT["scale_y"])
    else:
        # 카메라 상단에서 아래로 내려갈수록(physical_y 증가), 로봇 좌표값은 작아져야 함
        robot_y = CAM_TO_ROBOT["offset_y"] - (physical_y * CAM_TO_ROBOT["scale_y"])

    robot_z = CAM_TO_ROBOT["offset_z"]

    # (회전 로직은 복잡도를 줄이기 위해 단순 이동 후 필요시 적용)
    # 현재는 translation(평행이동)만 정확히 맞추는 것에 집중합니다.

    return [robot_x - 0.052 , robot_y + 0.025, robot_z, -180, 0, 180]

# 6) Pick 동작
def picknPlace(indy_t, pose, label):
    # 불량품인 경우에만 처리
    if label == "1 ng_w" or label == "3 ng_b":
        try:
            indy_t.connect()

            print(f"[PICK] 대상 위치: X={pose[0]:.4f}, Y={pose[1]:.4f}, Z={pose[2]:.4f}")

            # 접근 위치 (목표 위치 위 20mm)
            approach = pose.copy()
            approach[2] += 0.02  # 20mm 위

            # 파지 위치
            grasp = pose.copy()
            grasp[2] -= 0.09  # 30mm 아래 (물체 높이에 따라 조정)

            # 리트릿 위치 (파지 후 위로)
            retreat = approach.copy()

            # 불량품 배치 위치
            ng_position = [0.18632, 0.37008, 0.4, -180, 0, 180]
            ng_drop = ng_position.copy()
            ng_drop[2] -= 0.15  # 불량품 투하 위치

            indy_t.go_home()
            IsMoveDone(indy_t)

            indy_t.task_move_to(approach)
            IsMoveDone(indy_t)

            indy_t.task_move_to(grasp)
            IsMoveDone(indy_t)

            grip(True, indy_t)
            sleep(0.5)

            indy_t.task_move_to(retreat)
            IsMoveDone(indy_t)

            indy_t.task_move_to(ng_position)
            IsMoveDone(indy_t)

            indy_t.task_move_to(ng_drop)
            IsMoveDone(indy_t)

            grip(False, indy_t)
            sleep(0.5)

            indy_t.task_move_to(ng_position)
            IsMoveDone(indy_t)

            indy_t.go_home()
            IsMoveDone(indy_t)

            print("[PICK] 완료")

        except Exception as e:
            print(f"로봇 동작 중 오류: {e}")
        finally:
            indy_t.disconnect()
    else:
        print("정상품 - 로봇 동작 생략")

# 8) PLC 통신 함수
def plc_bitread(plc_t, address):
    try:
        bit_vals = plc_t.batchread_bitunits(address, 1)
        return bit_vals[0]
    except Exception as e:
        print(f"PLC 읽기 오류 ({address}): {e}")
        return 0

def plc_bitwrite(plc_t, address, value):
    try:
        plc_t.batchwrite_bitunits(address, [value])
    except Exception as e:
        print(f"PLC 쓰기 오류 ({address}): {e}")

# 9) 디버깅을 위한 시각화 함수
def visualize_detection(frame, box_rect, circles, robot_pose, label, conf):
    vis = frame.copy()

    # 고정 박스 표시
    if box_rect:
        x, y, w, h = box_rect
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 박스 중심선 표시
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.line(vis, (center_x, y), (center_x, y + h), (255, 0, 0), 1)
        cv2.line(vis, (x, center_y), (x + w, center_y), (255, 0, 0), 1)

        # 좌측 상단 표시 (offset 기준점)
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(vis, "Offset", (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 원형 물체 표시
        if circles:
            for i, circle in enumerate(circles):
                cx_img = x + circle["cx"]
                cy_img = y + circle["cy"]
                r = circle["r"]

                # 원 그리기
                color = (0, 0, 255) if i == 0 else (255, 0, 0)  # 첫번째 원은 빨간색
                cv2.circle(vis, (cx_img, cy_img), r, color, 2)
                cv2.circle(vis, (cx_img, cy_img), 2, (255, 255, 0), 3)

                # 좌표 표시
                cv2.putText(vis, f"{i+1}: ({circle['cx']}, {circle['cy']})",
                           (cx_img + 10, cy_img - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 물리적 좌표 계산 (디버깅용)
                if i == 0 and box_rect:
                    w_box, h_box = box_rect[2], box_rect[3]
                    phys_x, phys_y = pixel_to_physical(circle["cx"], circle["cy"], w_box, h_box)
                    cv2.putText(vis, f"Phys: ({phys_x:.4f}, {phys_y:.4f})",
                               (cx_img + 10, cy_img + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # 분류 결과 표시
    cv2.putText(vis, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Conf: {conf:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 로봇 좌표 표시
    if robot_pose:
        cv2.putText(vis, f"Robot X: {robot_pose[0]:.4f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(vis, f"Robot Y: {robot_pose[1]:.4f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(vis, f"Offset X: {CAM_TO_ROBOT['offset_x']:.4f}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"Offset Y: {CAM_TO_ROBOT['offset_y']:.4f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return vis

# 10) 테스트 함수 (캘리브레이션용)
def test_coordinate_conversion(cx_pixel, cy_pixel, box_width_pixel, box_height_pixel):
    """
    좌표 변환 테스트 함수
    """
    print(f"\n=== 좌표 변환 테스트 ===")
    print(f"입력 픽셀: ({cx_pixel}, {cy_pixel})")
    print(f"박스 크기: {box_width_pixel} x {box_height_pixel} 픽셀")

    # 픽셀 → 물리적 좌표
    phys_x, phys_y = pixel_to_physical(cx_pixel, cy_pixel, box_width_pixel, box_height_pixel)
    print(f"물리적 좌표: X={phys_x:.4f}m, Y={phys_y:.4f}m")

    # 물리적 → 로봇 좌표
    robot_pose = physical_to_robot(phys_x, phys_y)
    print(f"로봇 좌표: X={robot_pose[0]:.4f}, Y={robot_pose[1]:.4f}, Z={robot_pose[2]:.4f}")

    return robot_pose

# 11) 메인 함수
def main_function(indy_t, plc_t):
    stopsign = 0
    cap = cv2.VideoCapture(0)

    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("시스템 준비 완료. PLC 신호 대기 중...")
    print(f"Offset (좌측 상단): X={CAM_TO_ROBOT['offset_x']:.4f}, Y={CAM_TO_ROBOT['offset_y']:.4f}")
    print(f"박스 물리적 크기: {BOX_REAL_WIDTH}m x {BOX_REAL_HEIGHT}m")

    try:
        while True:
            camera_stop = plc_bitread(plc_t, "B101")

            if camera_stop == 1:
                print("\n=== 카메라 스톱퍼 동작 감지 ===")

                ret, frame = cap.read()
                if not ret:
                    print("카메라 캡처 실패")
                    continue

                # Step 1: 분류
                label, conf = classify_whole_image(frame)
                print(f"[CLASSIFY] → {label}, conf={conf:.3f}")

                # Step 2: 박스 ROI 추출
                roi, box_rect = detect_box(frame)
                if roi is None:
                    print("상자 ROI 추출 실패")
                    continue

                x_box, y_box, w_box, h_box = box_rect
                print(f"[BOX] 위치: ({x_box}, {y_box}), 크기: {w_box} x {h_box}")

                # Step 3: 원형 검출
                circles = detect_circles(roi)
                if not circles:
                    print("원형 물체 없음")
                    continue

                print(f"[CIRCLES] {len(circles)}개 검출됨")

                # 첫 번째(가장 큰) 원 선택
                target_circle = circles[0]
                print(f"[TARGET] ROI 내 좌표: cx={target_circle['cx']}, cy={target_circle['cy']}, r={target_circle['r']}")

                # 전체 이미지에서의 좌표 계산
                img_cx = x_box + target_circle["cx"]
                img_cy = y_box + target_circle["cy"]
                print(f"[TARGET] 전체 이미지 좌표: ({img_cx}, {img_cy})")

                # Step 4: 좌표 변환
                # ROI 내 좌표 → 물리적 좌표
                physical_x, physical_y = pixel_to_physical(
                    target_circle["cx"], target_circle["cy"], w_box, h_box
                )
                print(f"[PHYSICAL] 물리적 좌표: X={physical_x:.4f}m, Y={physical_y:.4f}m")

                # 물리적 좌표 → 로봇 좌표
                robot_pose = physical_to_robot(physical_x, physical_y)
                print(f"[ROBOT POSE] X={robot_pose[0]:.4f}, Y={robot_pose[1]:.4f}, Z={robot_pose[2]:.4f}")

                # 테스트 출력
                print(f"[OFFSET] 기준: X={CAM_TO_ROBOT['offset_x']:.4f}, Y={CAM_TO_ROBOT['offset_y']:.4f}")
                print(f"[DELTA] 변위: ΔX={physical_x:.4f}, ΔY={physical_y:.4f}")

                # Step 5: 시각화
                vis_frame = visualize_detection(frame, box_rect, [target_circle], robot_pose, label, conf)
                cv2.imshow("Detection", vis_frame)
                key = cv2.waitKey(1) & 0xFF

                # Step 6: PLC 신호 전송
                if label in ["1 ng_w", "3 ng_b"]:
                    while plc_bitread(plc_t, "B110") == 0 :
                        plc_bitwrite (plc_t, "B110", 1)
                    print("불량품 감지 - B110 신호 전송")
                else:
                    while plc_bitread(plc_t, "B112") == 0 :
                        plc_bitwrite (plc_t, "B112", 1)

                # Step 7: 로봇 스톱퍼 신호 대기
                print("[WAIT] 로봇 스톱퍼 신호 대기 중...")
                while True:
                    robot_stop = plc_bitread(plc_t, "B103")
                    emergency = plc_bitread(plc_t, "B106")

                    if emergency == 1:
                        print("정상품 출하")
                        stopsign = 1
                        break

                    if robot_stop == 1:
                        print("[ROBOT] 로봇 스톱퍼 동작 - 로봇 실행")

                        if label in ["1 ng_w", "3 ng_b"]:
                            picknPlace(indy_t, robot_pose, label)

                        # 완료 신호 전송
                        while plc_bitread(plc_t, "B111") == 0 :
                            plc_bitwrite (plc_t, "B111", 1)
                            sleep(0.5)
                        stopsign = 1
                        break

                    sleep(0.1)

                print("한 사이클 완료\n")

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("사용자에 의해 종료")
                break
            sleep(0.05)
            if stopsign :
                break
    except KeyboardInterrupt:
        print("프로그램 종료")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    robot_ip = "192.168.3.6"
    robot_name = "NRMK-Indy7"

    try:
        indy1 = client.IndyDCPClient(robot_ip, robot_name)
        print("로봇 클라이언트 생성 완료")
    except Exception as e:
        print(f"로봇 클라이언트 생성 실패: {e}")
        indy1 = None

    try:
        plc = mc.Type3E()
        plc.setaccessopt(commtype="binary")
        plc.connect("192.168.3.130", 1025)
        print("PLC 연결 성공")
    except Exception as e:
        print(f"PLC 연결 실패: {e}")
        plc = None

    if indy1 is not None and plc is not None:
        main_function(indy1, plc)

    if plc is not None:
        plc.close()
        print("PLC 연결 종료")