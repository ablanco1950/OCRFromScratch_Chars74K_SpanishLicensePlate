# -*- coding: utf-8 -*-
"""
Created on december 2022

@author Alfonsso Blanco García using
Chars74 Dataset: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

"""


import numpy as np

import os

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout


import cv2

def RecortaPorArriba(img):
    Altura=int(len(img)/2)
    while (Altura >= 0):
        SwHay=0
        for i in range(len(img[0])):
            if img[Altura][i]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[Altura:,:])
        Altura=Altura - 1
    return(img)
def RecortaPorAbajo(img):
    Altura=int(len(img)/2)
    while (Altura <= len(img)-1):
        SwHay=0
        for i in range(len(img[0])):
            if img[Altura][i]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[:Altura,:])
        Altura=Altura + 1
    return(img)

def RecortaPorDerecha(img):
    Anchura=int(len(img[0])/2)
    while (Anchura < len(img[0])-1):
        SwHay=0
        for i in range(len(img)):
            if img[i][Anchura]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[:,:Anchura])
        Anchura=Anchura +1
    return(img)
def RecortaPorIzquierda(img):
    Anchura=int(len(img[0])/2)
    while (Anchura >=0):
        SwHay=0
        for i in range(len(img)):
            if img[i][Anchura]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[:,Anchura:])
        Anchura=Anchura -1
    return(img)


dataset=[] 
labels=[]
count=0

folders=os.listdir("C:\\EnglishImgRedu\\EnglishImg\\English\\Img\\GoodImg\\Bmp")
for i in folders:
  
  Stri=str(i)
  
  
 
  for j in    os.listdir("C:\\EnglishImgRedu\\EnglishImg\\English\\Img\\GoodImg\\Bmp\\"+str(i)):
     
     im_new=cv2.imread("C:\\EnglishImgRedu\\EnglishImg\\English\\Img\\GoodImg\\Bmp\\"+str(i)+"\\"+str(j))
     im_new = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)
    
     ret2,im_new = cv2.threshold(im_new,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
     im_new=RecortaPorArriba(im_new)
     im_new=RecortaPorAbajo(im_new)
     
     im_new=RecortaPorDerecha(im_new)
     im_new=RecortaPorIzquierda(im_new)
     
     im_new=im_new/255.0
     resized_image = cv2.resize(im_new,(32,32),cv2.INTER_CUBIC)
    
     
     dataset.append(np.array([  np.asarray(resized_image) ] ))
     labels.append(count)
  count+=1


dataset = np.array(dataset)

dataset = np.array(dataset).reshape(5795,32,32)
labels = np.array(labels).reshape(-1,1)

# We will convert the labels into OneHotEncoding (Know more about that)
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

from sklearn.preprocessing import OneHotEncoder
type_encoder = OneHotEncoder()
labels=type_encoder.fit_transform(labels).toarray()

from keras.layers import   BatchNormalization, Activation

#https://keras.io/api/layers/initializers/
tensorflow.keras.initializers.Zeros()

# number of possible label values

nb_classes = 36

# Initialising the CNN
model = Sequential()

# 1 - Convolution

model.add(Conv2D(16,(3,3), padding='same', input_shape=(32, 32,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# 2nd Convolution layer

model.add(Conv2D(32,(3,3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# Flattening
model.add(Flatten())

# Big and decreasing denses mean better clasificacion in cascade

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# Train and save the model to use further

model.fit(dataset , labels, batch_size = 4, epochs = 240)
model.save("Model240Epoch.h5")
