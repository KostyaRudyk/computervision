import cv2
import numpy as np
image = cv2.imread("images/1.jpg")
print(image.shape)
# image= cv2.resize(image, (600, 600))
image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 100, 100)





kenel = np.ones((5,5), np.uint8)
print(kenel)
image = cv2.dilate(image, kenel, iterations=1)
image = cv2.erode(image, kenel, iterations=1)
cv2.imwrite("images/1.jpg", image)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()