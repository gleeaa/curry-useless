import cv2
from curry_utils import calculate_thickness_percentage

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thickness_percent, mask = calculate_thickness_percentage(frame)

        # Show results
        cv2.putText(frame, f"Thickness: {thickness_percent}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Curry View", frame)
        cv2.imshow("Curry Mask", mask)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
