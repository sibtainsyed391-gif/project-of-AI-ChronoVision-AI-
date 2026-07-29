import cv2
import sqlite3
from GUI import *

rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/TrainingData.yml")
cascadePath = "Classifiers/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

path='dataSet'

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="select * from People where id="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        nbr_predicted,conf=rec.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        profile=getProfile(nbr_predicted)
        if (profile!=None):
            cv2.putText(img, "Name: "+str(profile[1]),(x, y + h+ 30),font, 0.4, (0, 0, 255), 1)
            cv2.putText(img, "Age: "+str(profile[2]),(x, y + h + 50),font, 0.4, (0, 0, 255), 1)
            cv2.putText(img, "Gender: "+str(profile[3]),(x, y + h + 70),font, 0.4, (0, 0, 255), 1)
            cv2.putText(img, "Criminal Records: "+str(profile[4]),(x, y + h + 90),font, 0.4, (0, 0, 255), 1)
                        
        else:
            cv2.putText(img, "Name: Unknown", (x, y + h + 30), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Age: Unknown", (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Gender: Unknown", (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Criminal Records: Unknown", (x, y + h + 90), font, 0.4, (0, 0, 255), 1);

		
    cv2.imshow("img",img);
    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break;
    
cap.release()
cv2.destroyAllWindows()




