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
from libcamera import controls
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
BOX_WIDTH = 400
SHOW_ROI = False
SHOW_PROCESSED = True
LANG = "eng+tgl"
MSE_CUTOFF = 20000
# ----------------------------

ocr_result = {
    "text": "",
    "proc_time": 0.0,
    "timestamp": None,
    "error": None,
    "mse": 0.0,
}
ocr_queue = queue.Queue(maxsize=1)

preprocessed_img = None
blank = np.zeros(shape=[LINE_HEIGHT, BOX_WIDTH, 3], dtype=np.uint8)
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
        return None, image, None, None, None, time.time() - t0

    largest_contour = max(contours, key=cv2.contourArea)
    fingertip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    fx, fy = fingertip
    cv2.circle(image, fingertip, 12, (0, 0, 255), -1)

    y_end = max(fy - 5, 0)
    y_start = max(y_end - LINE_HEIGHT, 0)
    x_start = max(fx - (BOX_WIDTH // 2), 0)
    x_end = max(fx + (BOX_WIDTH // 2), 0)
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    crop = image[y_start:y_end, x_start:x_end]

    return crop if crop.size > 0 else None, image, fx, fy, x_start, time.time() - t0


def mse_diff(img_base, img_test):
    err = np.sum((img_base.astype("float") - img_test.astype("float")) ** 2)
    err /= float(img_base.shape[0] * img_test.shape[1])
    return err


def fast_diff(a, b):
    # Ensure same size
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[0], b.shape[1]), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 to avoid overflow
    diff = cv2.norm(a, b, cv2.NORM_L2)

    # Normalize by number of pixels to behave similar to MSE
    n = a.size
    return (diff * diff) / n


def ultra_fast_diff(a, b):
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[0], b.shape[1]), interpolation=cv2.INTER_LINEAR)

    return cv2.norm(a, b, cv2.NORM_L2)


def preprocess_for_ocr(image, fast=False):
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if fast:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binarized = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    else:
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


def run_ocr_select_word(img, finger_x):
    """
    Runs OCR once and selects the word whose bounding box
    horizontally overlaps the fingertip x-position.
    """
    t0 = time.time()
    config = r"--oem 3 --psm 7 -l " + LANG

    try:
        data = pytesseract.image_to_data(
            img,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        err = None
    except Exception as e:
        return "", str(e), time.time() - t0

    selected_word = ""
    min_dist = float("inf")

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue

        x = data["left"][i]
        w = data["width"][i]

        # Check if fingertip is inside this word's x-range
        if finger_x is not None and x <= finger_x <= x + w:
            selected_word = word
            return selected_word, err, time.time() - t0


         # FALLBACK ONLY if finger_x is unknown
        if finger_x is None:
            center_x = x + w / 2
            dist = abs(center_x - (img.shape[1] / 2))
            if dist < min_dist:
                min_dist = dist
                selected_word = word

    return selected_word, err, time.time() - t0



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
            processed, t_prep = preprocess_for_ocr(img)

            mse = ultra_fast_diff(prev_frame, preprocessed_img)
            if mse <= MSE_CUTOFF:
                finger_x = meta.get("finger_x", processed.shape[1] // 2)
                text, err, t_ocr = run_ocr_select_word(processed, finger_x)

            prev_frame = processed

            logger.info(f"OCR done frame {frame_id}: text='{text}'")

            if text and text != last_spoken_text:
                to_speak = text if len(text) < 200 else text[:200] + "..."
                logger.info(f"Saying with TTS: {to_speak}")
                tts.say(to_speak, wait4prev=True)
                last_spoken_text = text

            # t_total = time.time() - t_total_start
            t_total = t_prep + t_ocr
            ocr_result.update(
                {
                    "text": text,
                    "proc_time": t_total,
                    "timestamp": now(),
                    "error": err,
                }
            )

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
    picam2.set_controls(
        {
            "AfMode": controls.AfModeEnum.Continuous,
            "AfRange": controls.AfRangeEnum.Macro,
        }
    )

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

            crop, debug_img, fingertip_x, fingertip_y, crop_x_start, t_det = \
                detect_fingertip_and_crop_line(frame.copy())


            if crop is not None and (frame_count % OCR_FRAME_INTERVAL == 0):
                if ocr_queue.empty():

                    # Convert fingertip X from frame coords → crop-local coords
                    if fingertip_x is not None and crop_x_start is not None:
                        finger_x_crop = fingertip_x - crop_x_start
                    else:
                        finger_x_crop = None

                    meta = {
                        "frame_id": frame_count,
                        "finger_x": finger_x_crop,
                        "ts": now(),
                    }

                    ocr_queue.put_nowait((crop.copy(), meta))



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
                f"OCR proc_time: {ocr_result.get('proc_time', 0):.2f}s",
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
