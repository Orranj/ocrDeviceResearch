import cv2
import numpy as np
import pytesseract
from picamera2 import Picamera2
import threading
import queue
import time
from datetime import datetime

# ---------- Config ----------
CAM_SIZE = (800, 600)
FPS_TARGET = 10
OCR_FRAME_INTERVAL = 1      # attempt OCR every N frames
LINE_HEIGHT = 100           # crop height above fingertip
SIDE_CROP = 30              # left/right crop margin
SHOW_ROI = False             # show cropped ROI window
SHOW_PROCESSED = True       # show preprocessed window
LANG = 'eng+tgl'            # tesseract languages
# ----------------------------

# shared container for OCR result & debug
ocr_result = {
    'text': '',
    'proc_time': 0.0,
    'angle': 0.0,
    'timestamp': None,
    'error': None,
    'mse': 0.0,
}

# queue for images to be processed by OCR worker
ocr_queue = queue.Queue(maxsize=1)  # only keep 1 pending to avoid backlog

preprocessed_img = None
blank = np.zeros(shape=[100,740,3],dtype=np.uint8)
blank.fill(255)

prev_frame = blank

# ---------- Helper functions ----------
def now():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(msg):
    print(f"[{now()}] {msg}")

# Fingertip detection and crop
def detect_fingertip_and_crop_line(image):
    t0 = time.time()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # skin HSV range - may need tuning for lighting/skin tones
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, image, 0.0, time.time()-t0

    largest_contour = max(contours, key=cv2.contourArea)
    # top-most point = fingertip
    fingertip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    fx, fy = fingertip
    cv2.circle(image, fingertip, 12, (0, 0, 255), -1)

    # crop a strip above fingertip
    y_end = max(fy - 5, 0)
    y_start = max(y_end - LINE_HEIGHT, 0)
    x_start = SIDE_CROP
    x_end = image.shape[1] - SIDE_CROP
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    crop = image[y_start:y_end, x_start:x_end]

    return crop if crop.size > 0 else None, image, fy, time.time()-t0

# Fast deskew using minAreaRect
def correct_skew_fast(image):
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # invert threshold because we expect dark text on light background; adjust if opposite
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return image, 0.0, time.time()-t0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # convert OpenCV angle to deskew angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # WARNING THIS LINE IS USED TO MANUALLY OVERRIDE THE FUNCTION
    angle = 0

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle, time.time()-t0

def mse_diff(img_base, img_test):
    # img_ref is the base, img_test is what u compare it to
    err = np.sum((img_base.astype("float") - img_test.astype("float")) ** 2)
    err /= float(img_base.shape[0] * img_test.shape[1])
    
    return err

# Preprocessing (CLAHE + denoise + Otsu)
def preprocess_for_ocr(image):
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    # _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarized = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
    proc = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    global preprocessed_img
    preprocessed_img = proc
    return proc, time.time()-t0

# run tesseract
def run_ocr(img):
    t0 = time.time()
    config = r'--oem 3 --psm 7 -l ' + LANG
    try:
        text = pytesseract.image_to_string(img, config=config).strip()
        err = None
    except Exception as e:
        text = ""
        err = str(e)
    return text, err, time.time()-t0

# ---------- Worker thread ----------
def ocr_worker():
    log("OCR worker started")
    while True:
        try:
            item = ocr_queue.get()  # blocks until image available
            if item is None:
                log("OCR worker received shutdown signal")
                break
            img, meta = item  # meta can include frame id / timestamp
            frame_id = meta.get('frame_id', 0)

            log(f"OCR worker: received frame {frame_id} (queue size {ocr_queue.qsize()})")
            t_total_start = time.time()

            # deskew
            rotated, angle, t_deskew = correct_skew_fast(img)
            # preprocess
            processed, t_prep = preprocess_for_ocr(rotated)
            # OCR
            # text, err, t_ocr = run_ocr(processed)

            global prev_frame
            mse = mse_diff(prev_frame, preprocessed_img)
            if mse <= 18000:
                text, err, t_ocr = run_ocr(processed)
            prev_frame = processed

            t_total = time.time() - t_total_start
            ocr_result['text'] = text
            ocr_result['proc_time'] = t_total
            ocr_result['angle'] = angle
            ocr_result['timestamp'] = now()
            ocr_result['error'] = err
            # ocr_result['mse'] = mse

            log(f"OCR done frame {frame_id}: text='{text}'")
            log(f" timings (deskew={t_deskew:.3f}s, prep={t_prep:.3f}s, ocr={t_ocr:.3f}s) total={t_total:.3f}s angle={angle:.2f}°")
            log(f"MSE: {mse}")
            ocr_queue.task_done()

        except Exception as e:
            log(f"OCR worker exception: {e}")

# ---------- Main ----------
def main():
    picam2 = Picamera2()
    cam_config = picam2.create_video_configuration(
        main={"size": CAM_SIZE, "format": "RGB888"},
        controls={"FrameRate": FPS_TARGET}
    )
    picam2.configure(cam_config)
    picam2.start()

    # start OCR worker thread
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
            t0 = time.time()
            frame = picam2.capture_array()
            frames_in_sec += 1

            # fingertip detection
            crop, debug_img, fingertip_y, t_det = detect_fingertip_and_crop_line(frame.copy())
            # optionally show fingertip detection time in overlay

            # every N frames, if queue empty, send crop for OCR
            if crop is not None and (frame_count % OCR_FRAME_INTERVAL == 0):
                if ocr_queue.empty():
                    # put to queue (non-blocking) with meta
                    meta = {'frame_id': frame_count, 'ts': now()}
                    try:
                        ocr_queue.put_nowait((crop.copy(), meta))
                        last_queue_time = now()
                        log(f"Queued frame {frame_count} for OCR")
                    except queue.Full:
                        log("Queue full, skipping enqueue")

            # update overlay text if new result
            if ocr_result['timestamp'] is not None:
                displayed_text = ocr_result['text']

            # FPS calc every second
            if time.time() - last_fps_t >= 1.0:
                fps = frames_in_sec / (time.time() - last_fps_t + 1e-9)
                last_fps_t = time.time()
                frames_in_sec = 0
            else:
                fps = None

            # debug overlay
            overlay = displayed_text if displayed_text else "<no OCR yet>"
            y = 40
            cv2.putText(debug_img, f"OCR: {overlay}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y += 30
            qsize = ocr_queue.qsize()
            cv2.putText(debug_img, f"Queue: {qsize}  LastQ: {last_queue_time}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
            y += 25
            cv2.putText(debug_img, f"OCR proc_time: {ocr_result.get('proc_time',0):.2f}s angle: {ocr_result.get('angle',0):.1f}°", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            y += 25
            cv2.putText(debug_img, f"Fingertip detect: {t_det*1000:.0f}ms", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
            if fps is not None:
                cv2.putText(debug_img, f"FPS: {fps:.1f}", (10, y+28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # show preview
            cv2.imshow("Finger + Text Line (debug)", debug_img)

            # optionally show ROI / processed images (peek at windows)
            if SHOW_ROI and crop is not None:
                try:
                    cv2.imshow("ROI (cropped)", crop)
                except Exception:
                    pass

            # show last processed (we cannot access 'processed' directly here, so not shown).
            # if you want, store last processed image in shared var from worker.

            if SHOW_PROCESSED and crop is not None:
                try:
                    cv2.imshow("Preprocessed Image", preprocessed_img)
                except Exception:
                    pass
            
            frame_count += 1
            # exit on q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log("Quit requested")
                break

            # small sleep to yield CPU (camera capture already paced by picamera)
            # time.sleep(0.001)

    finally:
        # shutdown
        log("Shutting down: sending None to worker and waiting")
        try:
            # send shutdown signal
            ocr_queue.put_nowait(None)
        except Exception:
            pass
        worker.join(timeout=1.0)
        picam2.stop()
        cv2.destroyAllWindows()
        log("Exited cleanly")

if __name__ == "__main__":
    main()

