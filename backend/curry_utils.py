import cv2
import numpy as np

def calculate_thickness_percentage(frame):
    """
    Calculates a 'thickness percentage' for curry based on color and texture.
    Returns value between 0 and 100.
    """

    # Resize for faster processing
    frame = cv2.resize(frame, (320, 240))

    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Curry-like color range (yellow, orange, red)
    lower_curry = np.array([5, 50, 50])
    upper_curry = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_curry, upper_curry)

    # Remove noise
    mask = cv2.medianBlur(mask, 5)

    # Calculate percentage of curry pixels
    curry_area_ratio = np.sum(mask > 0) / mask.size

    # Texture analysis for thickness (less edges = thicker curry)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Invert edge density for thickness (more edges → watery, fewer edges → thick)
    thickness_score = (curry_area_ratio * 0.7 + (1 - edge_density) * 0.3) * 100

    return round(thickness_score, 2), mask
