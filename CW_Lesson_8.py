import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_casscade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')

face_net = cv2.dnn.readNetFromCaffe('dnl/deploy.prototxt', 'dnl/res10_300x100_smoothed_model.caffemodel')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() #ret-зчитаний або не зчитаний, фрейм- кадр
    if not ret:
        break
#_____________________________dnn______________________
    (h, w) = frame.shape[:2]#берігаємо тільки два значення
    blob = cv2.dnn.fromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))# формат фото
    face_net.setInput(blob)
    detections = face_net.forward()# пропускаємо кадри

    for i in range(detections.shape[2]):#роігаємо по кожному каналу
        confidence = detections[0, 0, i, 2]#впевненість мережі/

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (w, y ,y2, x2) = box.astype("int")
            x,y = max(0, x), max(0, y)
            x2,y2 = min(w-1,x2), min(y-1,y2)

            cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)

        cv2.imshow('DNN',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


#+++++++++++++++++++++++++++++cascades____________________

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)) #minneighbours - кількість опрацьованих сусідніх пікселів
    #
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)# перебираємо колжне фейсес і навколо нього малюмо прмрокутник
    # #roi - регіон оф інтерест область обличчя та область пікселів в якій знаходится обличчя
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #
    #     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1 ,minNeighbors=10, minSize=(10, 10)) #визначаємо очі
    #
    #
    #     for(ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255 ,0, 0 ), 1)
    #
    #     smile = smile_casscade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
    #     for(sx, sy, sw, sh) in smile:
    #         cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    #
    #
    #
    # cv2.putText(
    #     frame, f'Faces detected{len(faces)}',
    #     (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1
    # )
    # #
    # #
    # # cv2.imshow("Haar Face Tracking", frame)
    # # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
cv2.destroyAllWindows()

