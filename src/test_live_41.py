import logging
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import espeakng
import numpy as np
import pytesseract
from picamera2 import Picamera2

# ---------- Logging Setup ----------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=datetime.now().strftime("logs/finger_ocr_%Y-%m-%d_%H-%M-%S.log"),
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- Config ----------
CAM_SIZE = (800, 600)
FPS_TARGET = 10
OCR_FRAME_INTERVAL = 2
LINE_HEIGHT = 100
SIDE_CROP = 200
SHOW_ROI = False
SHOW_PROCESSED = True
LANG = "eng+tgl"
MSE_CUTOFF = 20000
# ----------------------------

ocr_result = {
    "text": "",
    "proc_time": 0.0,
    "angle": 0.0,
    "timestamp": None,
    "error": None,
    "mse": 0.0,
}
ocr_queue = queue.Queue(maxsize=1)

preprocessed_img = None
blank = np.zeros(shape=[LINE_HEIGHT, 800 - 2 * SIDE_CROP, 3], dtype=np.uint8)
blank.fill(255)
prev_frame = blank
tts = espeakng.Speaker()
tts.voice = "en-us"
tts.wpm = 140
last_spoken_text = ""
text = ""
err = 0
t_ocr = 0


def now():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg):
    logger.info(msg)


def detect_fingertip_and_crop_line(image):
    t0 = time.time()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, image, 0.0, time.time() - t0

    largest_contour = max(contours, key=cv2.contourArea)
    fingertip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    fx, fy = fingertip
    cv2.circle(image, fingertip, 12, (0, 0, 255), -1)

    y_end = max(fy - 5, 0)
    y_start = max(y_end - LINE_HEIGHT, 0)
    x_start = SIDE_CROP
    x_end = image.shape[1] - SIDE_CROP
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    crop = image[y_start:y_end, x_start:x_end]

    return crop if crop.size > 0 else None, image, fy, time.time() - t0


def correct_skew_fast(image):
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 50:
        return image, 0.0, time.time() - t0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    angle = 0  # override

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle, time.time() - t0


def mse_diff(img_base, img_test):
    err = np.sum((img_base.astype("float") - img_test.astype("float")) ** 2)
    err /= float(img_base.shape[0] * img_test.shape[1])
    return err


def fast_diff(a, b):
    # Ensure same size
    if a.shape != b.shape:
        return float("inf")

    # Convert to float32 to avoid overflow
    diff = cv2.norm(a, b, cv2.NORM_L2)

    # Normalize by number of pixels to behave similar to MSE
    n = a.size
    return (diff * diff) / n


def ultra_fast_diff(a, b):
    return cv2.norm(a, b, cv2.NORM_L2)


def preprocess_for_ocr(image):
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9
    )
    proc = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    global preprocessed_img
    preprocessed_img = proc
    return proc, time.time() - t0


def run_ocr(img):
    t0 = time.time()
    config = r"--oem 3 --psm 7 -l " + LANG
    try:
        text = pytesseract.image_to_string(img, config=config).strip()
        err = None
    except Exception as e:
        text, err = "", str(e)
    return text, err, time.time() - t0


def ocr_worker():
    logger.info("OCR worker started")
    global prev_frame, last_spoken_text, text, err, t_ocr

    while True:
        try:
            item = ocr_queue.get()
            if item is None:
                logger.info("OCR worker received shutdown signal")
                break

            img, meta = item
            frame_id = meta.get("frame_id", 0)
            logger.info(f"OCR worker: received frame {frame_id}")

            t_total_start = time.time()
            # rotated, angle, t_deskew = correct_skew_fast(img)
            processed, t_prep = preprocess_for_ocr(img)

            mse = ultra_fast_diff(prev_frame, preprocessed_img)
            if mse <= MSE_CUTOFF:
                text, err, t_ocr = run_ocr(processed)
            prev_frame = processed

            t_total = time.time() - t_total_start
            ocr_result.update(
                {
                    "text": text,
                    "proc_time": t_total,
                    "angle": -1,
                    "timestamp": now(),
                    "error": err,
                }
            )

            logger.info(f"OCR done frame {frame_id}: text='{text}'")

            if text and text != last_spoken_text:
                to_speak = text if len(text) < 200 else text[:200] + "..."
                logger.info(f"Saying with TTS: {to_speak}")
                tts.say(to_speak, wait4prev=True)
                last_spoken_text = text

            # log(
            #     f"timings (deskew={t_deskew:.3f}s, prep={t_prep:.3f}s, ocr={t_ocr:.3f}s) total={t_total:.3f}s angle={angle:.2f}°"
            # )
            logger.info(
                f"timings (prep={t_prep:.3f}s, ocr={t_ocr:.3f}s) total={t_total:.3f}s Error: {mse}"
            )

            ocr_queue.task_done()
        except Exception as e:
            logger.exception(f"OCR worker exception: {e}")


def main():
    picam2 = Picamera2()
    cam_config = picam2.create_video_configuration(
        main={"size": CAM_SIZE, "format": "RGB888"}, controls={"FrameRate": FPS_TARGET}
    )
    picam2.configure(cam_config)
    picam2.start()

    worker = threading.Thread(target=ocr_worker, daemon=True)
    worker.start()

    log("Starting main loop")
    frame_count = 0
    last_fps_t = time.time()
    frames_in_sec = 0
    displayed_text = ""
    last_queue_time = None

    try:
        while True:
            frame = picam2.capture_array()
            frames_in_sec += 1

            crop, debug_img, fingertip_y, t_det = detect_fingertip_and_crop_line(
                frame.copy()
            )

            if crop is not None and (frame_count % OCR_FRAME_INTERVAL == 0):
                if ocr_queue.empty():
                    meta = {"frame_id": frame_count, "ts": now()}
                    try:
                        ocr_queue.put_nowait((crop.copy(), meta))
                        last_queue_time = now()
                        logging.info(f"Queued frame {frame_count} for OCR")
                    except queue.Full:
                        logging.info("Queue full, skipping enqueue")

            if ocr_result["timestamp"] is not None:
                displayed_text = ocr_result["text"]

            if time.time() - last_fps_t >= 1.0:
                fps = frames_in_sec / (time.time() - last_fps_t + 1e-9)
                last_fps_t = time.time()
                frames_in_sec = 0
            else:
                fps = None

            overlay = displayed_text if displayed_text else "<no OCR yet>"
            y = 40
            cv2.putText(
                debug_img,
                f"OCR: {overlay}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            y += 30
            cv2.putText(
                debug_img,
                f"Queue: {ocr_queue.qsize()}  LastQ: {last_queue_time}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 128, 255),
                2,
            )
            y += 25
            cv2.putText(
                debug_img,
                f"OCR proc_time: {ocr_result.get('proc_time', 0):.2f}s angle: {ocr_result.get('angle', 0):.1f}°",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )
            y += 25
            cv2.putText(
                debug_img,
                f"Fingertip detect: {t_det * 1000:.0f}ms",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 255),
                2,
            )

            if fps is not None:
                cv2.putText(
                    debug_img,
                    f"FPS: {fps:.1f}",
                    (10, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Finger + Text Line (debug)", debug_img)

            if SHOW_ROI and crop is not None:
                try:
                    cv2.imshow("ROI (cropped)", crop)
                except Exception:
                    pass

            if SHOW_PROCESSED and crop is not None:
                try:
                    cv2.imshow("Preprocessed Image", preprocessed_img)
                except Exception:
                    pass

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested")
                break

    finally:
        logger.info("Shutting down: sending None to worker and waiting")
        try:
            ocr_queue.put_nowait(None)
        except Exception:
            pass
        worker.join(timeout=1.0)
        picam2.stop()
        cv2.destroyAllWindows()
        logger.info("Exited cleanly")


if __name__ == "__main__":
    main()
