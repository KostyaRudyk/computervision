import cv2
import numpy as np

me = cv2.imread('images/me.jpg', 1)
cv2.rectangle(me, (750, 950), (400, 450), (0, 255, 0), 3)
cv2.putText(me, "Rudyk Kostyantyn", (400, 1000), cv2.FONT_ITALIC, 1,(0, 255, 0))



cv2.imwrite('images/me_result.jpg', me)
cv2.imshow('me', me)
cv2.waitKey(0)
cv2.destroyAllWindows()
