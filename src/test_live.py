import cv2
from picamera2 import Picamera2
import arfred_code

# Initialize OpenCV window
cv2.startWindowThread()

# Initialize Picamera2 and configure the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 960)}))
picam2.start()

while True:
    # Capture frame-by-frame
    im = picam2.capture_array()

    debug, process = arfred_code.ocr_pipeline(im)

    # Display the resulting frame
    cv2.imshow("debug", debug)
    cv2.imshow("Camera", process)
   
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cv2.destroyAllWindows()
