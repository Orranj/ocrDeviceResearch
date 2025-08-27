import cv2
import numpy as np
import pytesseract
# from pytesseract import Output
import time
# from picamera2 import Picamera2, Preview


# --- Lighting & Contrast Enhancement ---
def enhance_contrast_and_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in np.arange(256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, look_up_table)
    return enhanced

# --- Deskew ---
def deskew_using_osd(img):
    try:
        osd = pytesseract.image_to_osd(img)
        rotation_angle = int([line.split(": ")[1] for line in osd.split("\n") if "Rotate" in line][0])
        if rotation_angle != 0:
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w // 2, h // 2), -rotation_angle, 1)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(e)
    return img

# --- OCR with Bounding Boxes ---
def bounding_boxes_and_text(img):
    print("text detection")
    start_time = time.time()
    config = r'--oem 1 --psm 7'
    text = pytesseract.image_to_string(img, config=config)
    return img, text
    # data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    # print("data extraction finished", time.time() - start_time)
    # start_detect = time.time()
    # detected_text = []
    # for i, word in enumerate(data['text']):
    #     if word.strip() and int(data['conf'][i]) > 50:
    #         word_start = time.time()
    #         detected_text.append(word)
    #         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    #         # print(f"detected {word} in {time.time() - word_start}")
    # print("word detection finished", time.time() - start_detect)
    # return img, " ".join(detected_text)

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
        print("No finger detected.")
        return None, image

    largest_contour = max(contours, key=cv2.contourArea)
    fingertip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    fx, fy = fingertip
    cv2.circle(image, fingertip, 20, (0, 0, 255), -1)

    line_height = 250  # height of text
    side_crop = 700 # amount of px to crop off from the sides
    shift_bottom = 10 # shift the crop box vertically downwards by this many px
    y_start = max(fy - line_height, 0)
    y_end = max(fy + shift_bottom, 0)
    image = cv2.rectangle(image, (side_crop, y_start), (image.shape[1] - side_crop, y_end), (0, 255, 0), 2)
    crop = image[y_start:y_end, side_crop:image.shape[1] - side_crop]

    if crop.size == 0:
        print("Crop resulted in empty image.")
        return None, image
     
    return crop, image

# --- Preprocessing for OCR ---
def process(img):
    start_time = time.time()
    max_dim = 700
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    print("resized", time.time() - start_time)
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # img = enhance_contrast_and_lighting(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.fastNlMeansDenoising(img)
    print("denoise", time.time() - start_time)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    print("binarize", time.time() - start_time)
    # _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # white_ratio = np.mean(otsu) / 255
    # if white_ratio > 0.85 or white_ratio < 0.15:
    #     otsu = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY, 31, 3)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    # otsu = deskew_using_osd(otsu)
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result_img, text = bounding_boxes_and_text(color_img)
    print("bounding boxes and yext", time.time() - start_time)
    return result_img, text


# will work on a video version later
# TODO: finish doig this function
# def take_pic():
#     picam2 = Picamera2()
#     camera_config = picam2.create_still_configuration(main={"size": (640, 480)})
#     picam2.configure(camera_config)
#     picam2.start_preview(Preview.QTGL)
#     picam2.start()
#     time.sleep(5)


# --- Main ---
def main():
    # Tk().withdraw()
    # image_path = filedialog.askopenfilename(
    #     title="Select an image",
    #     filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    # )
    # if not image_path:
    #     print("No file selected.")
    #     return

    start_time = time.time()
    orig_img = cv2.imread("test.jpg")
    if orig_img is None:
        print("Error: Could not read image.")
        return

    # deskewed_img = deskew_using_osd(orig_img)
    # print("Deskewed image")
    cropped_line, debug_img = detect_fingertip_and_crop_line(orig_img)
    print(f"Image cropped to finger level: {time.time() - start_time:.2f}")
    if cropped_line is None or cropped_line.size == 0:
        print("Skipping OCR because no valid crop was found.")
        return

    print("\nProcessing cropped line above fingertip...")
    processed_img, text = process(cropped_line)
    print(f"Detected Text:\n{text}")
    print(f"Total Processing time: {time.time() - start_time:.2f}")

    # cv2.imshow("Finger Detection", debug_img)
    # cv2.imshow("Cropped Line", cropped_line)
    # cv2.imshow("Processed Line", processed_img)
    cv2.imwrite("outputs/arfred_debug.jpg", debug_img)
    cv2.imwrite("outputs/arfred_process.jpg", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
