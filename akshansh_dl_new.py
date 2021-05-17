# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:48:20 2020

@author: Dell
"""
import keras as ks
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD
from keras.activations import *
import time
from sklearn.model_selection import train_test_split
def mydata_div(X,Y,test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test

def my_tensorboard(a):
    NAME=a+"BCI_MI{}".format(int(time.time()))
    tensorboard=ks.callbacks.TensorBoard(log_dir='logs{}/'.format(NAME))
    return tensorboard

def mycon1d(X_train,Y_train,X_test,Y_test,batch_size,epochs,verbose):
    c=Y_train.shape[1]
    # n=x.shape[1]
    # inp=tf.keras.Input(shape=(n,1))
    model=ks.Sequential()
    model.add(ks.layers.convolutional.Conv1D(16,kernel_size=5,strides=1,input_shape=(X_train.shape[1:3])))
    model.add(ks.layers.convolutional.Conv1D(32,kernel_size=5,strides=1))
    model.add(ks.layers.convolutional.MaxPooling1D(pool_size=2))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dropout(.02))
    model.add(ks.layers.Dense(c,activation='softmax'))
    # model.add(ks.activations('softmax'))
    opt=SGD(lr=.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history=model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test,Y_test), verbose = verbose,shuffle=True)
    return history

def mysimpleconv1d(X_train,Y_train,X_test,Y_test,batch_size,epochs,verbose):
    model = ks.Sequential()
    my_filters=32
    my_kernel_size = 1
    my_strides = 1
    model.add(ks.layers.convolutional.Conv1D(my_filters, my_kernel_size, input_shape = X_train.shape[1:3], padding='same', strides = my_strides))
    model.add(ks.layers.Dense(1000,activation='relu'))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(Y_train.shape[1],activation='softmax'))
    # model.add(ks.activations('softmax'))
    print(model.summary())
    startlearningrate=0.001
    adam = ks.optimizers.Adam(lr=startlearningrate)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    # callbacks=all_callbacks(stop_patience=1000,
    #                         lr_factor=0.5,
    #                         lr_patience=10,
    #                         lr_epsilon=0.000001,
    #                         lr_cooldown=2,
    #                         lr_minimum=0.0000001)
    history=model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test,Y_test), verbose = verbose,shuffle=True)
    return history



