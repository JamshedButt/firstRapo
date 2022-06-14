import cv2

face = cv2.CascadeClassifier("C:\\Users\\Jamshed Butt\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier("C:\\Users\\Jamshed Butt\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
  ret,frame =  cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  faces = face.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize = (30,30),
      flags = cv2.CASCADE_SCALE_IMAGE)
  

      

  for (x,y,w,h) in faces:
    cv2.rectangle(frame , (x,y), (x+w, y+h), (0,255,0), 2)
    roi_gray = gray[y:y+h , x:x+w]
    roi_color = frame[y:y+h , x:x+w]
    
    eyes = eye.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color , (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)  

  cv2.imshow("Video",frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()