import cv2
import numpy as np
    from numpy.ma.core import filled

    img = np.zeros((800,512,3), np.uint8)

    # img[100:150,200:280] = 152,8,235
    # img[:] = 152,8,235

    cv2.rectangle(img,(100,100),(200,200),(255,201,25),1)
    cv2.line(img,(200,100),(300,150),(255,201,25),67)
    print(img.shape)
    cv2.line(img, (0, img.shape[0] // 2),(0,img.shape[1]//2),(255,201,25),67)
    cv2.circle(img, (200,200),40,(255,255,0),-1)
    cv2.putText(img, "name",(100,400), cv2.FONT.ITALIC, 1)








    cv2.imshow('KOTAXBAC', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()