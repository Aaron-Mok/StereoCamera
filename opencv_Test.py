import cv2

img = cv2.imread('window.png')
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(cv2.cuda.getCudaEnabledDeviceCount())
print(cv2.getBuildInformation())