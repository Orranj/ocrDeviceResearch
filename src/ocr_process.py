# imports needed
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# remove this if used in raspi
#from google.colab.patches import cv2_imshow

# --- Finger Detection/Cropping Functions ---

def finger_detect(img):
    pass

# --- Preprocessing Functions ---

def deskew_using_osd(img):
    try:
        osd = pytesseract.image_to_osd(img)
        # Parse the rotation angle
        rotation_angle = int([line.split(": ")[1] for line in osd.split("\n") if "Rotate" in line][0])

        if rotation_angle != 0:
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w // 2, h // 2), -rotation_angle, 1)
            img = cv2.warpAffine(img, m, (w, h))
            print(f"Deskewed using OSD. Rotation angle: {rotation_angle}")
        else:
            print("No deskewing needed. Rotation angle: 0")
    except Exception as e:
        print("Deskewing failed:", e)

    return img

def bounding_boxes(img):
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    text = pytesseract.image_to_string(img)
    print("Detected Text:\n", text)

    for i, word in enumerate(data['text']):
        if word.strip() and int(data['conf'][i]) > 60:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return img, text

def process(img, display_invert = False):
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize image
    # img = cv2.resize(img, (0, 0), 2, 0.12, 0.1)

    # adjust contrast and sharpness
    # img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    # denoise image
    img = cv2.fastNlMeansDenoising(img)

    # binarize image
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)

    # invert for easier OCR
    img = cv2.bitwise_not(img)

    # correcting skew/rotation using OSD (orientation and script detection)
    img = deskew_using_osd(img)

    # rotating image
    img = cv2.rotate(img, cv2.ROTATE_180)

    # turn to RGB for colored bounding boxes
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # display bounding boxes
    img, text = bounding_boxes(img)

    if not display_invert:
        # if display_invert = True, the display image is inverted
        img = cv2.bitwise_not(img)

    return img, text

def main():
    img_name = input("Name of image to be processed: ")

    # gets filename of uploaded file
    # REPLACE THIS LINE IN RASPBERRY PI
    orig_img = cv2.imread(f"src/{img_name}.png")

    # Show original image
    # print("Original Image:")
    # smol_img = cv2.resize(orig_img, (0, 0), 2, 0.12, 0.12)
    cv2.imshow("Original Image", orig_img)

    # Processed image and text output
    print("\nprocessing image:")
    processed_img, text = process(orig_img, display_invert=False)

    # Resize for display
    target_width = 500
    h, w = processed_img.shape[:2]
    scale = target_width / w
    display_img = cv2.resize(processed_img, (int(w * scale), int(h * scale)))
    cv2.imshow("Processed Image", processed_img)
    cv2.imwrite(f"src/{img_name}_process.png", processed_img)
if __name__ == "__main__":
    main()
