import cv2
import numpy as np

def get_curry_percentage(frame):
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for curry (adjust these values based on lighting)
    lower_curry = np.array([10, 80, 80])   # Example lower HSV bound
    upper_curry = np.array([30, 255, 255]) # Example upper HSV bound

    # Create mask
    mask = cv2.inRange(hsv, lower_curry, upper_curry)

    # Calculate percentage of curry color in frame
    curry_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    percentage = (curry_pixels / total_pixels) * 100

    return percentage, mask
