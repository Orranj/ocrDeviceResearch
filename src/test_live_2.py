import cv2
import numpy as np
import pytesseract
from picamera2 import Picamera2


# --- Fingertip detection and crop line above ---
def detect_fingertip_and_crop_line(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, image

    # fingertip = top-most point of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    fingertip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    fx, fy = fingertip
    cv2.circle(image, fingertip, 20, (0, 0, 255), -1)

    # crop a strip above fingertip
    line_height = 100  # adjust depending on font size
    side_crop = 30
    y_end = max(fy - 5, 0)
    y_start = max(y_end - line_height, 0)
    x_start = side_crop
    x_end = image.shape[1] - side_crop
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    crop = image[y_start:y_end, x_start:x_end]

    return crop if crop.size > 0 else None, image

# --- OCR ---
def run_ocr(img):
    config = r'--oem 3 --psm 7 -l eng+tgl'   # single line mode
    text = pytesseract.image_to_string(img, config=config).strip()
    return text

def main():
    picam2 = Picamera2()
    cam_config = picam2.create_video_configuration(
        main={"size": (800, 600), "format": "RGB888"},
        controls={"FrameRate": 30}  # Set desired framerate
    )
    picam2.configure(cam_config)
    picam2.start()

    print("Press 'q' to quit.")
    while True:
        frame = picam2.capture_array()

        cropped_line, debug_img = detect_fingertip_and_crop_line(frame)

        detected_text = ""
        if cropped_line is not None:
            # preprocess for OCR
            gray = cv2.cvtColor(cropped_line, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 9, 9)
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            detected_text = run_ocr(color_img)
            # detected_text = ""

        # display text overlay
        cv2.putText(debug_img, detected_text, (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        cv2.imshow("Finger + Text Line", debug_img)
        # if cropped_line is not None:
            # cv2.imshow("Cropped Line", cropped_line)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
