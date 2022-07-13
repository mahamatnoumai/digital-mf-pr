import cv2
import numpy as np
import os 
from time import perf_counter 
import requests
from datetime import datetime
import csv
import time

def openDoor():
                
    def trigger(folder,value1):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
        photoFile="log/"+ folder +"/Photo"+ str(dt_string) +".jpg"
        cv2.imwrite(photoFile, img)
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        return photoFile,dt_string
        
    def write_to_csv(photoFile,dt_string,name):
        with open("log/log_Files.csv", mode="a") as csv_readings:
            logfile_write = csv.writer(csv_readings, delimiter=",", quotechar="/", quoting=csv.QUOTE_MINIMAL)
            write_to_log = logfile_write.writerow([photoFile,dt_string,name])
        return(write_to_log)
        
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    doorIsOpen=False
    batsoi=False
    id = 0
    cnt=0
    names = ['None', 'Ali'] 

    cam = cv2.VideoCapture(1)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img =cam.read()
        #img = cv2.flip(img, 0) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
            )
            
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                cnt=cnt+1;
                if cnt%5==0:
                    doorIsOpen=True
                    photoFile,dt_string= trigger("Success","The door would be open!!!!")
                    write_to_csv(photoFile,dt_string,str(id))
                    return doorIsOpen,batsoi
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                cnt=cnt+1;
                if cnt%50==0:
                    batsoi=True
                    photoFile,dt_string=trigger("Fail","Breach attempt coming up, cops..!!")
                    write_to_csv(photoFile,dt_string,str(id))
                    return doorIsOpen,batsoi
                    

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
            
    return doorIsOpen,batsoi
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
        

isDoorOpen=False
batsoi=False
#doorQuest = input('Do you want to open the door?[y/n]')
#if doorQuest == 'y':
isDoorOpen,batsoi = openDoor()
if isDoorOpen==True:
    print('Face Recognized!! The Door is Open!!!')
    print(isDoorOpen,batsoi)
  
if batsoi==True:
    print('is it open')
    print(isDoorOpen)


#else:
    #print('???')
