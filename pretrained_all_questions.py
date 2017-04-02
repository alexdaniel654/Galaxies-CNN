import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from skimage.io import imread
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



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
    nb_epoch = 30
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
    #Y = np.zeros([nim,(Gtruth.shape[1]-2)*4])
    #Y = np.matlib.repmat(Gtruth[0:Gtruth.shape[0],2:Gtruth.shape[1]],4,1)
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
    
    model.load_weights('gal.hdf5')

# rules for saving the model to disk
#    model_checkpoint = ModelCheckpoint('gal.hdf5', monitor = 'val_loss',\
#                                   save_best_only = True)
#
#    print('Commencing model training...')
#    hist = model.fit(X_train, Y_train,
#              batch_size=batch_size,
#              nb_epoch=nb_epoch,
#              validation_data=(X_test, Y_test),
#                              shuffle=True,
#                              callbacks=[model_checkpoint])    
#%%
    evalu = model.evaluate(X_test, Y_test, batch_size=32)
    p=model.predict(X_test, batch_size=batch_size)
    pclasses = model.predict_classes(X_train, batch_size=batch_size)
    #pproba = model.predict_proba(X_test, batch_size=batch_size)
#%%    
    RMSE = mean_squared_error(Y_test, p)**0.5
    print('RMSE = ',RMSE)
    
    np.save('pred',p)
    
#%%
#    imind = np.zeros(p.shape[1])
#    imcat = np.zeros([p.shape[1], 424, 424, 3])
#    for n in np.arange(p.shape[1]):
#        iminds = np.where(pclasses==n)
#        if iminds[0].shape[0] == 0:
#            imcat[n,:,:,:] = np.ones([424,424,3])*255
#        else:
#            imind = iminds[0][0]
#            im = imread(imloc[imind])
#            imcat[n,:,:,:] = (im)
#                 
#    x = np.mean(imcat, axis=1)
#    x = np.mean(x, axis=1)
#    x = np.mean(x, axis=1)
#    np.save('imcat', imcat)
#    
##%%
#
#for n in np.arange(imcat.shape[0]):
#    plt.imshow(np.abs(imcat[n,:,:,:]-255))
#    plt.axis('off')
#    fname = 'imcat'+str(n)+'.png'
#    plt.savefig(fname, dpi=600)
#%%
import glob
imloc = glob.glob('D:/Alex Daniel/OwnCloud/University/Image Processing/Term 2/Galaxies/Training Images/*.jpg')
for n in np.arange(pclasses.shape[0]):
    im = imread(imloc[n+ntrain+1])
    