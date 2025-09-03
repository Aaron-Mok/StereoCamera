from picamera2 import Picamera2
import cv2

picamR = Picamera2(0)
# config = picam2.create_preview_configuration(main={"format": "RGB888"})
config = picamR.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}) 
picamR.configure(config)
picamR.start()


picamL = Picamera2(1)
# config = picam2.create_preview_configuration(main={"format": "RGB888"})
config = picamL.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}) 
picamL.configure(config)
picamL.start()


while True:
    frameR = picamR.capture_array()
    frameR = cv2.flip(frameR, -1) 
    cv2.imshow("Right Camera", frameR)

    frameL = picamL.capture_array()
    frameL = cv2.flip(frameL, -1)
    cv2.imshow("Left Camera", frameL)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()