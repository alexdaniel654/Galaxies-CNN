# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:00:01 2017

@author: Alex Daniel
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def imsplit(imlocation, trainprop):
    imo = np.load(imlocation)
    nim = imo.shape[0]    
    imotrain = imo[0:round(trainprop*nim), :,:,:]
    imotest  = imo[round(trainprop*nim):nim,:,:,:]
    return imotrain, imotest
    
    
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config = config) as s:

    batch_size = 2**7
    nb_classes =159
    nb_epoch = 90
    data_augmentation = False
    trainprop=0.95

    Gtruth = pd.read_csv('train_reform.csv',',')
    Gtruth = Gtruth.as_matrix()
    X_train, X_test = imsplit('img_64_60000.npy', trainprop)
    ntrain = X_train.shape[0]
    ntest  = X_test.shape[0]
    nim    = ntrain+ntest
#%%
    Y = np.zeros([nim,Gtruth.shape[1]-2])
    Y = Gtruth[0:nim,2:Gtruth.shape[1]]
#    Y = np.zeros([nim,(Gtruth.shape[1]-2)*4])
#    Y = np.matlib.repmat(Gtruth[0:Gtruth.shape[0],2:Gtruth.shape[1]],4,1)
    Y_train = Y[0:ntrain]
    Y_test = Y[ntrain:nim]
        
#%%
    model = Sequential()
    
    model.add(Convolution2D(32, 7, 7, border_mode='same', input_shape=X_train.shape[1:], name='conv1'))
    model.add(Activation('relu', name='act_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool_1'))
    model.add(Convolution2D(64, 5, 5, border_mode='same', name='conv2'))
    model.add(Activation('relu', name='act_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool_2'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv3'))
    model.add(Activation('relu', name='act_3'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv4'))
    model.add(Activation('relu', name='act_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool_4'))
    model.add(Flatten())
    model.add(MaxoutDense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
# let's train the model using SGD.
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255


# rules for saving the model to disk
    model_checkpoint = ModelCheckpoint('gal_long.hdf5', monitor = 'val_loss',\
                                   save_best_only = True)

    print('Commencing model training...')
    hist = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
                              shuffle=True,
                              callbacks=[model_checkpoint])    

    p=model.predict(X_test, batch_size=batch_size)
#%%    
    RMSE = mean_squared_error(Y_test, p)**0.5
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig('loss.png', dpi=600)
    plt.show()
    
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.grid()
    plt.savefig('acc.png', dpi=600)
    plt.show()
    
    print('RMSE = ',RMSE)