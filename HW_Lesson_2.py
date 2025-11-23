import cv2
import numpy as np

me = cv2.imread('images/me.jpg', 0)
me = cv2.resize(me,(me.shape[1]//2, me.shape[0]//2))
me = cv2.Canny(me, 100, 200)


sign = cv2.imread('images/sign1.jpg', 0)
sign =  cv2.resize(sign,(sign.shape[1]//3, sign.shape[0]//3))
sign = cv2.Canny(sign, 255,255)

cv2.imwrite("images/me.jpg", me)
cv2.imwrite('images/sign1.jpg', sign)
cv2.imshow('sign', sign)
cv2.imshow('me', me)
cv2.waitKey(0)
cv2.destroyAllWindows()