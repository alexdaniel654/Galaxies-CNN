# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:18:06 2017

@author: Alex Daniel
"""
import glob
import skimage.io
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imresize
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

def imloadcube(train):
    dim=2**7
    imloc = glob.glob('D:/Alex Daniel/OwnCloud/University/Image Processing/Term 2/Galaxies/Galaxies/TrainingSub/*.jpg')
    #imloc = glob.glob('D:/Alex Daniel/OwnCloud/University/Image Processing/Term 2/Galaxies/Training Images/*.jpg')
    nim = len(imloc)
    imo = np.zeros([nim, dim, dim, 3])
    gid = np.zeros(nim)
    n=0
    for filename in imloc:
        gid[n] = int(filename[len(filename)-10:len(filename)-4])
        im=skimage.io.imread(filename)
        imo[n,:,:,:]=imresize(im, (dim,dim))
        n=n+1
        
    imotrain = imo[0:round(train*nim), :,:,:]
    imotest  = imo[round(train*nim):nim,:,:,:]
    gidtrain = gid[0:round(train*nim)]
    gidtest  = gid[round(train*nim):nim]
    
    return imotrain, gidtrain, imotest, gidtest

def imsplit(imlocation, trainprop):
    imo = np.load(imlocation)
    nim = imo.shape[0]    
    imotrain = imo[0:round(trainprop*nim), :,:,:]
    imotest  = imo[round(trainprop*nim):nim,:,:,:]
    return imotrain, imotest
    
    
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config = config) as s:



    Gtruth = pd.read_csv('training_solutions_rev1.csv',',')

#with tf.Session() as sess:
#    with tf.device("/cpu:0"):

    batch_size = 2**7
    nb_classes =3
    nb_epoch = 30
    data_augmentation = False
    trainprop=0.95


    

#        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#    X_train, Gid_train, X_test, Gid_test = imloadcube(trainprop)
    X_train, X_test = imsplit('img_64_60000.npy', trainprop)
    ntrain = X_train.shape[0]
    ntest  = X_test.shape[0]
    nim    = ntrain+ntest
    print(ntrain, 'training images')
    print(ntest, 'test images')

    Y = np.zeros([nim,3])
    Y[:,0] = Gtruth['Class1.1'][0:nim]
    Y[:,1] = Gtruth['Class1.2'][0:nim]
    Y[:,2] = Gtruth['Class1.3'][0:nim]
    Y_train = Y[0:ntrain]
    Y_test = Y[ntrain:nim]
        
    #Y_train = Gtruth[0:ntrain].as_matrix()
    #Y_test  = Gtruth[ntrain:nim].as_matrix()
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
    model_checkpoint = ModelCheckpoint('gal.hdf5', monitor = 'val_loss',\
                                   save_best_only = True)

    print('Commencing model training...')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
                              shuffle=True,
                              callbacks=[model_checkpoint])    

    p=model.predict(X_test, batch_size=batch_size)
    