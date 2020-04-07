import cv2
cap = cv2.VideoCapture("C:/Users/15064/Desktop/猛龙过江.mp4")
success, frame = cap.read()
index = 1 
while success :
         index = index+1
         cv2.imwrite(str(index)+".png",frame)
         if index > 20:
                  break;
         success,frame  = cap.read()    
cap.release()
