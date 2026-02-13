import cv2
import numpy as np #математична бібліотека що працює з масивами чисел

caption = cv2.VideoCapture(0)

lower_red1 = np.array([0,100,100])
upper_red1 = np.array([10,255,255])

lower_red2 = np.array([160,100,100])
upper_red2 = np.array([180,255,255])

points =
while True:
    ret, frane = cap.read# ret= зчитаний чи не зчитаний кадр, а фрейм сам кадр
    if not ret:
        break
    
    
    frame= cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    mask1=cv2. inRange(hsv, lower_red1, upper_red1)
    mask2=cv2. inRange(hsv, lower_red2, upper_red2)

    mask = cv2.biteise_or(mask1, mask2)
    
    
    
    contours, _ = cv2.findCountours(mask, cv2.RETR_EXTERNAL, cv2,CHAIN_APPROX_SIMPLE)
    
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            M= cv2. moments(cnt)
            if m['m00'] !=0:
                cx = int(M['m10']/)
                cy = int(M['m01'])
                
                cv2.circle
                points.append((cx,cy))
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    
                    cv2.line(frame, points[i-1])
    
    
    
    cv2.imshow("video", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.DestroyAllWindows()