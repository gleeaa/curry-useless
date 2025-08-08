import cv2
from curry_thickness import get_curry_percentage

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    percentage, mask = get_curry_percentage(frame)

    # Show percentage on video
    cv2.putText(frame, f"Curry Thickness: {percentage:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("Curry Thickness Detector", frame)
    cv2.imshow("Mask", mask)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
