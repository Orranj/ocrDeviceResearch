# test_ocr.py
import pytesseract
from PIL import Image

image = Image.open('sample_image.jpg')  
text = pytesseract.image_to_string(image)  
print(text)