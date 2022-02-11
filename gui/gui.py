from tkinter import *
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile, Image, ImageTk
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import time
root=Tk()
root.geometry('675x450')
bg = PhotoImage(file = "home.png") 
label1 = Label( root, image = bg)
label1.place(x = 0, y = 0)
root.resizable(False,False)
root.resizable(False,False)
cwd = os.getcwd()

def train(x1,x2,x3,x4,x5,x6,x7,x8,x9,train_progress):
    train_progress.configure(text='Training...')
    os.chdir(cwd)
    CLASS_NAMES=os.listdir('data/')
    print(CLASS_NAMES)
    train_datagen = ImageDataGenerator(rescale= 1./255,height_shift_range=x5,zoom_range=x3,width_shift_range=x4,rotation_range=x8,shear_range=x2,horizontal_flip=x1,brightness_range=[0.4,1.5])
    training_set = train_datagen.flow_from_directory(r'data',target_size=(64,64),batch_size=32,class_mode='categorical')
    validation_datagen =ImageDataGenerator(rescale= 1./255)
    validation_set=validation_datagen.flow_from_directory(r'validation_data',target_size=(64,64),batch_size=16,class_mode='categorical')
    model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(CLASS_NAMES),activation='softmax')
    ])
    opt=RMSprop(lr=x6)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    history=model.fit(training_set,steps_per_epoch=len(training_set),epochs=x7,validation_data=validation_set,validation_steps=len(training_set)/2)
    model.save(x9+'.h5')
    train_progress.configure(text='Training done')
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    print(np.mean(val_acc))
    print(np.mean(acc))
    epochs = range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def add():
    os.chdir(cwd)
    window=Toplevel()
    window.geometry('675x450')
    global bg3
    bg3 = PhotoImage(file = "add.png")
    label1 = Label( window, image = bg3)
    label1.place(x = 0, y = 0)
    window.resizable(False,False)
    
    
    def add_functionality():
        vid=cv2.VideoCapture(0)
        cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        name=entry1.get()
        os.mkdir(r'data/'+name)
        os.chdir(r'data/'+name)
        count=0
        validation_count=0
        validation_init=True
        while(True):
            _,frame=vid.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=cascade.detectMultiScale(gray,1.1,5)
            if count==178:
                    if validation_init==True:
                        cv2.putText(frame, 'Press v to add validation data', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                        cv2.imshow("HI",frame)
                        if cv2.waitKey(1) & 0xFF == ord('v'):
                            validation_init=False
                            os.chdir(cwd)
                            os.mkdir(r'validation_data/'+name)
                            os.chdir(r'validation_data/'+name)
                    else:
                        if validation_count==88:
                            break
                        else:
                            for face in faces:
                                if(face[3]>100 and face[2]>100):                
                                    cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                                    cv2.imwrite(str(validation_count)+'.png',frame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]])
                                    validation_count=validation_count+1
                            cv2.imshow("HI",frame)
            else:
                for face in faces:
                    if(face[3]>100 and face[2]>100):                
                        cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                        cv2.imwrite(str(count)+'.png',frame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]])
                        count=count+1
                cv2.imshow("HI",frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()
        #train()
        window.destroy()
    
    Button(window,text=' ADD ',padx=70,pady=15,command=add_functionality,fg='white',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=250,y=390)
    Label(window,text="ADD NEW USER",bg='black',fg='white',font=('Comic Sans MS',20)).place(x=230,y=30)
    Label(window,text="Enter user name to add",bg='black',fg='white',font=('Arial',12)).place(x=15,y=352)
    entry1=Entry(window,width=15,font='Calibri 20')
    entry1.insert(0,'')
    entry1.place(x=230,y=352,height=30)
def recognize():
    window=Toplevel()
    window.geometry('675x450')
    global bg1
    bg1 = PhotoImage(file = "recognize.png")
    label1 = Label( window, image = bg1)
    label1.place(x = 0, y = 0)
    window.resizable(False,False)
    
    
    def vid():
        vid=cv2.VideoCapture(0)
        cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        facemodel = tf.keras.models.load_model(entry1.get()+'.h5')

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
                            print(os.listdir('data/')[x])
                            cv2.putText(frame, os.listdir('data/')[x], (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                            break
                    else:
                        cv2.putText(frame,"Unknown user", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2, cv2.LINE_AA)
                            
                    print(result)
            cv2.imshow("HI",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                window.destroy()
                break
        vid.release()
        cv2.destroyAllWindows()
    Label(window,text="RECOGNIZE  FACE",bg='black',fg='white',font=('Comic Sans MS',20)).place(x=100,y=10)
    Label(window,text="Enter name of model",bg='black',fg='white',font=('Arial',12)).place(x=50,y=160)
    Button(window,text=' RECOGNIZE ',padx=70,pady=15,command=vid,fg='black',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=50,y=250)
    entry1=Entry(window,width=15,font='Calibri 20')
    entry1.insert(0,'')
    entry1.place(x=50,y=190,height=30)
def optimize():
    os.chdir(cwd)
    window=Toplevel()
    window.geometry('675x450')
    global bg2
    bg2 = PhotoImage(file = "optimize.png")
    label1 = Label( window, image = bg2)
    label1.place(x = 0, y = 0)
    window.resizable(False,False)
    
    
    def optimize_functionality():
        vid=cv2.VideoCapture(0)
        cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        print('The users present are',os.listdir('data/'))
        name=entry1.get()
        count=max(list(map(lambda x: int(x[:len(x)-4]),os.listdir('data/'+name+'/'))))+1
        initial=max(list(map(lambda x: int(x[:len(x)-4]),os.listdir('data/'+name+'/'))))+1
        os.chdir(r'data/'+name)
        while(True):
            _,frame=vid.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=cascade.detectMultiScale(gray,1.1,5)
            for face in faces:
                cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                cv2.imwrite(str(count)+'.png',frame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]])
                count=count+1
            cv2.imshow("HI",frame)
            if count==initial+178:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()
        #train()
    Button(window,text=' OPTIMIZE ',padx=70,pady=15,command=optimize_functionality,fg='black',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=40,y=300)
    Label(window,text="OPTIMIZE FACE",bg='black',fg='white',font=('Comic Sans MS',20)).place(x=250,y=10)
    Label(window,text="Enter name you want to optimize",bg='black',fg='white',font=('Arial',10)).place(x=40,y=160)
    entry1=Entry(window,width=15,font='Calibri 20')
    entry1.insert(0,'')
    entry1.place(x=40,y=200,height=30)
def remove():
    os.chdir(cwd)
    window=Toplevel()
    window.geometry('675x450')
    global bg4
    bg4 = PhotoImage(file = "remove.png")
    label1 = Label( window, image = bg4)
    label1.place(x = 0, y = 0)
    window.resizable(False,False)
    
    
    def remove_func():
        shutil.rmtree(r'data/'+entry1.get())
        os.chdir(cwd)
        shutil.rmtree(r'validation_data/'+entry1.get())
        window.destroy()
    Button(window,text=' REMOVE ',padx=70,pady=15,command=remove_func,fg='black',bg='#FF0000',borderwidth=0,activeforeground='white',activebackground='#8B0000').place(x=230,y=390)
    Label(window,text="REMOVE USER",bg='black',fg='white',font=('Comic Sans MS',20)).place(x=230,y=30)
    Label(window,text="Enter user name to remove",bg='black',fg='white',font=('Arial',12)).place(x=15,y=352)
    entry1=Entry(window,width=12,font='Calibri 20')
    entry1.insert(0,'')
    entry1.place(x=235,y=352,height=30)

def train_help():
    window=Toplevel()
    window.geometry('675x450')
    global bg5
    bg5 = PhotoImage(file = "help.png")
    label1 = Label( window, image = bg5)
    label1.place(x = 0, y = 0)
    window.resizable(False,False)
    horizontal_flip_frame=Frame(window,width=250,height=50,bg='black')
    horizontal_flip_frame.place(x=325,y=50)
    Label(horizontal_flip_frame,bg='black',fg='white',text="Flips a random no of trained images horizontal\nly to generate more unique images").place(x=0,y=0)
    rotational_range_frame=Frame(window,width=250,height=50,bg='black')
    rotational_range_frame.place(x=325,y=100)
    Label(rotational_range_frame,bg='black',fg='white',text="Randomly rotates trained images taking range\n of the rotation in degrees as a parameter").place(x=0,y=0)
    shear_range_frame=Frame(window,width=250,height=50,bg='black')
    shear_range_frame.place(x=325,y=150)
    Label(shear_range_frame,bg='black',fg='white',text="Radomly pulls the image in the opposite sides \nshear range in percentage as a parameter").place(x=0,y=0)
    zoom_range_frame=Frame(window,width=250,height=50,bg='black')
    zoom_range_frame.place(x=325,y=200)
    Label(zoom_range_frame,bg='black',fg='white',text="Randomly zooms trained images taking \nrange of the zoom in percentage as a parameter").place(x=0,y=0)
    width_shift_frame=Frame(window,width=250,height=50,bg='black')
    width_shift_frame.place(x=325,y=250)
    Label(width_shift_frame,bg='black',fg='white',text="Randomly horizontally moves trained images.\nshift in percentage as a parameter").place(x=0,y=0)
    height_shift_frame=Frame(window,width=250,height=50,bg='black')
    height_shift_frame.place(x=325,y=300)
    Label(height_shift_frame,bg='black',fg='white',text="Randomly vertically moves trained images.\nshift in percentage as a parameter").place(x=0,y=0)
    lr_frame=Frame(window,width=250,height=50,bg='black')
    lr_frame.place(x=325,y=350)
    Label(lr_frame,bg='black',fg='white',text="The amount the neural network changes its\nelf after every image.The smaller lr the lesser \nvariation in accuracy but requires more epochs").place(x=0,y=0)
    epochs_frame=Frame(window,width=250,height=50,bg='black')
    epochs_frame.place(x=325,y=400)
    Label(epochs_frame,bg='black',fg='white',text="No of times the neural network goes overall \nthe trained images").place(x=0,y=0)
    
    Label(window,text='Horizontal flip:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=50)
    Label(window,text='Rotation range:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=100)
    Label(window,text='Shear range:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=150)
    Label(window,text='Zoom range:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=200)
    Label(window,text='Width shift range:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=250)
    Label(window,text='Height shift range:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=300)
    Label(window,text='Learning rate:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=350)
    Label(window,text='Epochs:',fg='white',bg='black',font=('Arial',15)).place(x=10,y=400)


def train_win():
    window=Toplevel()
    window.geometry('675x450')
    global bg3
    bg3 = PhotoImage(file = "train.png")
    label1 = Label( window, image = bg3)
    label1.place(x = 0, y = 0)
    window.resizable(False,False)
    
    horizontal_flip=IntVar()
    shear_range=Entry(window,width=33)
    zoom_range=Entry(window,width=33)
    width_shift_range=Entry(window,width=33)
    height_shift_range=Entry(window,width=33)
    learning_rate=Entry(window,width=33)
    epochs_input=Entry(window,width=33)
    rotation_range=Entry(window,width=33)
    model_name=Entry(window,width=33)
    train_progress=Label(window,text='',bg='white')
    train_progress.place(x=551,y=575)
    horizontal_button_yes=Radiobutton(window,text="Yes",variable=horizontal_flip,value=1,bg='white')
    horizontal_button_no=Radiobutton(window,text="No",variable=horizontal_flip,value=2,bg='white')
    Button(window,text='?',command=train_help,padx=10,pady=10,fg='white',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=650,y=1)
    Button(window,text=' Train ',command=lambda x1=horizontal_flip,x2=shear_range,x3=zoom_range,x4=width_shift_range,x5=height_shift_range,x6=learning_rate,x7=epochs_input,x8=rotation_range,x9=model_name,x10=train_progress:train(bool(x1.get()),int(x2.get())*0.01,int(x3.get())*0.01,int(x4.get())*0.01,int(x5.get())*0.01,float(x6.get()),int(x7.get()),int(x8.get())*0.01,x9.get(),x10),padx=70,pady=15,fg='white',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=250,y=395)
    Label(window,text="TRAIN FACES",bg='black',fg='white',font=('Comic Sans MS',20)).place(x=200,y=20)
    
    Label(window,text='Shear range:',bg='black',fg='white').place(x=50,y=160)
    Label(window,text='Zoom range:',bg='black',fg='white').place(x=400,y=160)
    Label(window,text='Width shift range:',bg='black',fg='white').place(x=50,y=220)
    Label(window,text='Height shift range:',bg='black',fg='white').place(x=400,y=220)
    Label(window,text='Learning rate:',bg='black',fg='white').place(x=50,y=280)
    Label(window,text='Epochs:',bg='black',fg='white').place(x=400,y=280)
    Label(window,text='Horizontal flip:',bg='black',fg='white').place(x=50,y=100)
    Label(window,text='Rotation range:',bg='black',fg='white').place(x=400,y=100)
    Label(window,text='Name of model:',bg='black',fg='white').place(x=50,y=340)
    Label(window,text='Label list:',bg='black',fg='white').place(x=400,y=340)
    Label(window,text=str(os.listdir('data/')),bg='white',fg='black').place(x=400,y=370)
    horizontal_button_yes.place(x=50,y=130)
    horizontal_button_no.place(x=100,y=130)
    
    shear_range.insert(0,'20')
    shear_range.place(x=50,y=190)
    zoom_range.insert(0,'0')
    zoom_range.place(x=400,y=190)
    width_shift_range.insert(0,'15')
    width_shift_range.place(x=50,y=250)
    height_shift_range.insert(0,'0')
    height_shift_range.place(x=400,y=250)    
    learning_rate.insert(0,'0.001')
    learning_rate.place(x=50,y=310)
    epochs_input.insert(0,'10')
    epochs_input.place(x=400,y=310)
    rotation_range.insert(0,'45')
    rotation_range.place(x=400,y=130)
    model_name.insert(0,'3face11(1)')
    model_name.place(x=50,y=370)
    


Button(root,text='RECOGNIZE FACE',padx=70,pady=15,command=recognize,fg='black',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=50,y=140)
Button(root,text=' OPTIMIZE FACE  ',padx=70,pady=15,command=optimize,fg='black',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=50,y=200)
Button(root,text='   TRAIN MODEL  ',padx=70,pady=15,command=train_win,fg='black',bg='#007BFF',borderwidth=0,activeforeground='white',activebackground='#0069D9').place(x=50,y=260)
Button(root,text=' ADD NEW USER  ',padx=70,pady=15,command=add,fg='black',bg='#20B2AA',borderwidth=0,activeforeground='white',activebackground='#3CB371').place(x=50,y=320)
Button(root,text=' REMOVE   USER  ',padx=70,pady=15,command=remove,fg='black',bg='#FF0000',borderwidth=0,activeforeground='white',activebackground='#8B0000').place(x=50,y=380)




#train()
