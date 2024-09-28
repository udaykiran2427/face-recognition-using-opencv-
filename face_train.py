import cv2 as cv
import numpy as np
import os 

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ["Tom curise","Zyan"]
DIR = r'D:\Numpy\images_for_training'

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            Face_rect = haar_cascade.detectMultiScale(gray,5,4)

            for (x,y,w,h) in Face_rect:
                face_roi = gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)

                
create_train()


features = np.array(features,dtype='object')
labels = np.array(labels)

face_recognization = cv.face.LBPHFaceRecognizer_create()
face_recognization.train(features,labels)
face_recognization.save('face_reco_model.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)

