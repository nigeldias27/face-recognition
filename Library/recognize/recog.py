import random
import numpy as np
import tensorflow as tf
import os
import cv2
def vid(image,name_of_model,label_list,cwdir):
    if image!='video':
        os.chdir(cwdir)
        cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
        facemodel = tf.keras.models.load_model(name_of_model+'.h5')
        img=cv2.imread(image)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray,1.1,5)
        for face in faces:
                    if(face[3]>100 and face[2]>100):
                        cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                        test_image=cv2.resize(img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]],(64,64))
                        test_image=np.expand_dims(test_image,axis=0)
                        result=facemodel.predict(test_image/255)
                        for x in range(len(result[0])):
                            if result[0][x]>0.8:
                                cv2.putText(img,label_list[x], (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                                return label_list[x],img
                        cv2.putText(img,"Unknown user", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                        return "Unknown user",img

    else:
        os.chdir(cwdir)
        vid=cv2.VideoCapture(0)
        cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        facemodel = tf.keras.models.load_model(name_of_model+'.h5')

        while(True):
            _,frame=vid.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=cascade.detectMultiScale(gray,1.1,5)
            for face in faces:
                if(face[3]>100 and face[2]>100):
                    cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                    test_image=cv2.resize(frame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]],(64,64))
                    test_image=np.expand_dims(test_image,axis=0)
                    result=facemodel.predict(test_image/255)
                    for x in range(len(result[0])):
                        if result[0][x]>0.8:
                            #print(label_list[x])
                            cv2.putText(frame, label_list[x], (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                            return label_list[x],frame
                    cv2.putText(frame,"Unknown user", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                    return "Unknown user",frame        
                    print(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()
