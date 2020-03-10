# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:25:29 2020

@author: IMPOSSIBLE
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')


X_train = X_train/255
X_test = X_test/255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
nClasses = Y_test.shape[1]

def model():
    model = Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(nClasses,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = model()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test), epochs=15, batch_size=200,verbose=1)
scores = model.evaluate(X_test,Y_test,verbose=0)
print("CNN error: %.2f%%" %(100-scores[1]*100))


