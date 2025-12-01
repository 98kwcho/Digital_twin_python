import cv2
import numpy as np

img = cv2.imread("0.jpg")
orig = img.copy()

# === STEP 1: 상단 밝은 영역 제거 ===
h, w = img.shape[:2]

# 상단 25% 지우기 (너 이미지 기준 최적값)
cut_top = int(h * 0.25)
mask_img = img[cut_top:h, 0:w].copy()


# === STEP 2: Grayscale + Blur ===
gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)

# === STEP 3: Adaptive threshold (어두운 상자만 강조) ===
th = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    35, 5
)

# === STEP 4: Morphological closing ===
kernel = np.ones((7,7), np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)


# === STEP 5: Contour 탐색 ===
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_area = 0
best_box = None

for cnt in contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    area = w_box * h_box
    
    # 상자처럼 충분히 큰 면적만
    if area < 5000:
        continue

    # 직사각형 비율 필터 (너 상자 비율 기준)
    aspect = w_box / float(h_box)
    if 1.2 < aspect < 3.5:  # 대략적인 상자 형태 필터
        if area > best_area:
            best_area = area
            best_box = (x, y, w_box, h_box)


if best_box is None:
    raise Exception("⚠ 상자를 자동 검출하지 못했습니다.")

# === STEP 6: mask_img에서의 ROI → 원본 이미지 좌표로 보정 ===
x, y, w_box, h_box = best_box
y = y + cut_top  # 상단 자른 부분 다시 보정

roi = orig[y:y+h_box, x:x+w_box]

# === 디버깅용 표시 ===
debug = orig.copy()
cv2.rectangle(debug, (x,y), (x+w_box,y+h_box), (0,255,0), 3)

cv2.imshow("threshold", th)
cv2.imshow("debug_box", debug)
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
