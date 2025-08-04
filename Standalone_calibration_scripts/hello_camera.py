from picamera2 import Picamera2
import cv2

picam2 = Picamera2(0)
# config = picam2.create_preview_configuration(main={"format": "RGB888"})
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}) 
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1) 
    cv2.imshow("Pi Camera ISP Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()