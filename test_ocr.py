import cv2
import os
import subprocess
import time

print("? Live preview running in a separate window...")
print("??  Press ENTER in this terminal to capture an image.")

# Run libcamera-hello in the background (preview-only)
subprocess.Popen("libcamera-hello -t 0", shell=True)

# Wait for user keypress to trigger capture
input("? Press ENTER to capture and display...")

# Kill preview (or it might conflict with capture)
subprocess.run("pkill libcamera-hello", shell=True)

# Wait a bit before capturing to avoid conflicts
time.sleep(1)

# Take the image (no preview)
subprocess.run("libcamera-still -o captured2.jpg --nopreview", shell=True)

# Load and show in OpenCV
img = cv2.imread("captured2.jpg")
if img is not None:
    cv2.imshow("? Captured Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image capture failed!")
