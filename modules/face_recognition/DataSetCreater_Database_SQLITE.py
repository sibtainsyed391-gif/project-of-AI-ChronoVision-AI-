import cv2
import numpy as np
import sqlite3

detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap=cv2.VideoCapture(0);

def insertOrUpdate(id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="select * from People where Id="+str(id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(id)
    else:
        cmd="INSERT INTO people(ID,Name) Values("+str(id)+",' "+str(Name)+" ' )"
    conn.execute(cmd)
    conn.commit()

id=input('enter user id')
Name=input('enter user Name')
insertOrUpdate(id,Name)
sampleNum=0;
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100);
    cv2.imshow('frame',img);
    cv2.waitKey(100);
    if(sampleNum>50):
        break    
cap.release()
cv2.destroyAllWindows()



