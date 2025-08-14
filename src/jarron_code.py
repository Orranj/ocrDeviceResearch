import cv2
import numpy as np
import pytesseract
import time
import os

# --- Conditional Deskew ---
def deskew_using_osd(img):
    try:
        osd = pytesseract.image_to_osd(img)
        rotation_angle = int(next(
            line.split(": ")[1] for line in osd.split("\n") if "Rotate" in line
        ))
        if rotation_angle not in (0, 180):
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w // 2, h // 2), -rotation_angle, 1)
            img = cv2.warpAffine(img, m, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    except:
        pass
    return img

# --- Conditional Enhancement ---
def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.std() < 40:  # Low contrast
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 9
        )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# --- Fingertip Detection & Dynamic Crop ---
def detect_fingertip_and_crop(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, image

    largest_contour = max(contours, key=cv2.contourArea)
    fingertip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    cv2.circle(image, fingertip, 15, (0, 0, 255), -1)

    # Dynamic crop height: detect text region above fingertip
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if text_contours:
        # Find contour(s) closest to fingertip y
        closest_contour = min(text_contours, key=lambda c: abs(cv2.boundingRect(c)[1] - fingertip[1]))
        x, y, w, h = cv2.boundingRect(closest_contour)
        y_start = max(y - 5, 0)
        y_end = min(y + h + 5, image.shape[0])
    else:
        # Fallback to fixed height
        y_start = max(fingertip[1] - 250, 0)
        y_end = min(fingertip[1] + 10, image.shape[0])

    side_crop = 700
    crop = image[y_start:y_end, side_crop:image.shape[1] - side_crop]

    return crop if crop.size > 0 else None, image

# --- OCR ---
def run_ocr(img):
    return pytesseract.image_to_string(img, config='--oem 1 --psm 7').strip()

# --- Main Pipeline ---
def main():
    os.makedirs("outputs", exist_ok=True)
    start_time = time.time()

    img = cv2.imread("test.jpg")
    if img is None:
        print("Error: Could not read image.")
        return

    # 1. Resize
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Deskew (only if needed)
    img = deskew_using_osd(img)

    # 3. Enhance (optional if needed before crop)
    img = enhance_for_ocr(img)

    # 4. Crop dynamically to line height
    cropped, debug_img = detect_fingertip_and_crop(img)
    if cropped is None:
        print("No fingertip/line detected.")
        return

    # 5. OCR
    text = run_ocr(cropped)

    # Debug output
    cv2.imwrite("outputs/debug_finger.jpg", debug_img)
    cv2.imwrite("outputs/cropped_line.jpg", cropped)

    print(f"Detected Text: {text}")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()

