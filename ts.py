import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# =========================
# 설정
# =========================
IMAGE_PATH = "0.jpg"          # 테스트할 이미지
MODEL_PATH = "keras_Model.h5"    # 분류 모델
LABEL_PATH = "labels.txt"        # 라벨 파일 (0 ok, 1 ng_x 이런 형식)

# =========================
# 모델 로드
# =========================
print("[INFO] loading model...")
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABEL_PATH, "r").readlines()


# =========================
# 1. 상자 자동 검출 함수
# =========================
def detect_box(image):
    """
    이미지에서 상자를 자동으로 찾아 ROI와 상자 좌표를 반환
    return: roi, (x, y, w, h)
    """
    orig = image.copy()
    h, w = image.shape[:2]

    # 상단 밝은 영역 (조명/레일) 조금 잘라내기
    cut_top = int(h * 0.20)
    work = image[cut_top:h, 0:w].copy()

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 상자 내부(어두운 영역) 강조
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    kernel = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best_box = None

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        if area < 5000:  # 너무 작은 것은 무시
            continue

        aspect = w_box / float(h_box)
        # 대략 가로가 세로의 1.2~3.5배 사이면 상자로 판단
        if 1.2 < aspect < 3.5:
            if area > best_area:
                best_area = area
                best_box = (x, y, w_box, h_box)

    if best_box is None:
        raise RuntimeError("상자를 찾지 못했습니다.")

    x, y, w_box, h_box = best_box
    # 상단 잘라낸 만큼 y 보정
    y = y + cut_top

    # 박스 가장자리 여유분 조금 안쪽으로 줄이기 (레일 영역 제거용)
    margin_x = int(w_box * 0.05)
    margin_y = int(h_box * 0.08)
    x_in = max(0, x + margin_x)
    y_in = max(0, y + margin_y)
    w_in = max(1, w_box - margin_x * 2)
    h_in = max(1, h_box - margin_y * 2)

    roi = orig[y_in:y_in + h_in, x_in:x_in + w_in]

    return roi, (x_in, y_in, w_in, h_in), th, orig


# =========================
# 2. ROI 내 원형 물체 + 분류
# =========================
def detect_and_classify_objects(roi):
    """
    상자 ROI 안에서 원형 물체를 찾고,
    각각을 분류 모델에 넣어 O/X 판정
    """
    result = roi.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=80,
        param2=30,
        minRadius=20,
        maxRadius=60
    )

    detections = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for (cx, cy, r) in circles[0, :]:
            # 원 영역 crop
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(roi.shape[1], cx + r), min(roi.shape[0], cy + r)
            obj = roi[y1:y2, x1:x2]

            # 분류 모델 입력 형태로 변환
            pil_img = Image.fromarray(obj).convert("RGB")
            pil_img = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)

            data = np.asarray(pil_img).astype(np.float32)
            data = (data / 127.5) - 1
            data = np.expand_dims(data, axis=0)

            pred = model.predict(data)
            idx = int(np.argmax(pred))
            label = class_names[idx].strip()
            conf = float(pred[0][idx])

            detections.append({
                "cx": int(cx),
                "cy": int(cy),
                "r": int(r),
                "label": label,
                "conf": conf
            })

            # 시각화
            color = (0, 255, 0) if "o" in label.lower() else (0, 0, 255)
            cv2.circle(result, (cx, cy), r, color, 3)
            cv2.circle(result, (cx, cy), 2, (0, 255, 255), 3)
            cv2.putText(
                result,
                f"{label} {conf:.2f}",
                (cx - 40, cy - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    else:
        print("[INFO] 원형 물체를 찾지 못했습니다.")

    return result, detections


# =========================
# 메인 테스트
# =========================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    # 1) 상자 자동 검출
    roi, box_rect, th, debug_img = detect_box(img)
    x, y, w_box, h_box = box_rect
    print(f"[INFO] Box ROI: x={x}, y={y}, w={w_box}, h={h_box}")

    # 디버그용 박스 표시
    cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)

    # 2) ROI 안에서 원형 물체 + 분류
    roi_result, detections = detect_and_classify_objects(roi)

    # 3) 결과 출력
    print("=== DETECTIONS ===")
    for d in detections:
        print(f"center=({d['cx']},{d['cy']}), r={d['r']}, label={d['label']}, conf={d['conf']:.3f}")

    cv2.imshow("threshold (for box)", th)
    cv2.imshow("debug_box", debug_img)
    cv2.imshow("ROI (box only)", roi)
    cv2.imshow("ROI result (objects + labels)", roi_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
