# -*- coding: utf-8 -*-

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

with tf.Session() as sess:
    with tf.device("/cpu:0"):

        batch_size = 32
        nb_classes = 10
        nb_epoch = 10
        data_augmentation = False

# input image dimensions
        img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
        img_channels = 3

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:], name='conv1'))
        model.add(Activation('relu', name='act_1'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool_1'))
        model.add(Convolution2D(64, 3, 3, border_mode='same',
                        input_shape=(16,16,32), name='conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool_2'))
        model.add(Flatten())
        model.add(Dense(512))
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
        model_checkpoint = ModelCheckpoint('cifar10_cnn.hdf5', monitor = 'val_loss',\
                                   save_best_only = True)

        print('Commencing model training...')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                                  shuffle=True,
                                  callbacks=[model_checkpoint])

