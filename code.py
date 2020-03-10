# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:11:29 2020

@author: IMPOSSIBLE
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical

(X_train,Y_train), (X_test,Y_test) = mnist.load_data()

#Testing the dataset
"""
for i in range(1,5):
    plt.subplot(1,4,i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    print(Y_train[i])
plt.show()
"""
nPixels = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],nPixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],nPixels).astype('float32')

#Normalizing the inputs to bring into the range of 0-1

X_train = X_train/255
X_test = X_test/255

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
nClasses = Y_test.shape[1]

def model():
    model = Sequential()
    model.add(Dense(nPixels,input_dim=nPixels, kernel_initializer='normal',activation='relu'))
    model.add(Dense(2*nPixels, kernel_initializer='normal',activation='relu'))
    model.add(Dense(392,kernel_initializer='normal',activation='relu'))
    model.add(Dense(nClasses,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model

model = model()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=50,batch_size=100,verbose=1)
scores = model.evaluate(X_test,Y_test,verbose=1)
print("Error : %.2f%%" % (100-scores[1]*100))



