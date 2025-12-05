import cv2
import numpy as np
NG = cv2.imread('images/Chokopay.jpg')

scale = 3
NG =  cv2.resize(NG, (NG.shape[1] * scale, NG.shape[0] * scale))

NG_copy = NG.copy()

NG = cv2.cvtColor(NG, cv2.COLOR_BGR2GRAY) # перевели в градацію сірого
NG = cv2.GaussianBlur(NG, (1, 1), 0) #БЛЮР
NG = cv2.equalizeHist(NG)

NG_edges = cv2.Canny(NG, 1, 1)# виявлення країв

countours, hierarchy = cv2.findContours(NG_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cunt in countours:
    area = cv2.contourArea(cunt)
    if area>100:
        x, y, w, h = cv2.boundingRect(cunt)

        cv2.drawContours(NG_copy,[cunt], -1, (0, 255, 0), 2)

        cv2.rectangle(NG_copy, (x, y), (x + w, y + h), (0, 255, 0), 2) #

        text_y = y - 10 if y - 10 > 20 else y + 10

        text = f'x:{x}, y:{y}, S:{int(area)}'
        cv2.putText(NG_copy, text,(x,text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




cv2.imshow('NG', NG)
cv2.imshow('NG_copy', NG_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
