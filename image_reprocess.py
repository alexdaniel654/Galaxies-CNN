# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:36:42 2017

@author: Alex Daniel
"""
import numpy as np
import skimage
from skimage.io import imread
from scipy.misc import imresize
import glob


dim=424#2**6
totim = 100#61578
#imloc = glob.glob('D:/Alex Daniel/OwnCloud/University/Image Processing/Term 2/Galaxies/Galaxies/TrainingSub/*.jpg')
imloc = glob.glob('D:/Alex Daniel/OwnCloud/University/Image Processing/Term 2/Galaxies/Training Images/*.jpg')
nim = len(imloc)
imo = np.zeros([totim*4, dim, dim, 3])
n=0
aug = 0

#for filename in imloc:
#    while n<totim:
if aug == 1:
    while n<totim:
        im=imread(imloc[n])
        imcrop = im[round(im.shape[0]/4):round(3*im.shape[0]/4), round(im.shape[1]/4):round(3*im.shape[1]/4)]
        imc1 = imcrop[0:round(2*imcrop.shape[0]/3), 0:round(2*imcrop.shape[1]/3)]
        imc2 = imcrop[round(1*imcrop.shape[0]/3):round(3*imcrop.shape[0]/3), 0:round(2*imcrop.shape[1]/3)]
        imc3 = imcrop[0:round(2*imcrop.shape[0]/3), round(1*imcrop.shape[1]/3):round(3*imcrop.shape[1]/3)]
        imc4 = imcrop[round(1*imcrop.shape[0]/3):round(3*imcrop.shape[0]/3), round(1*imcrop.shape[1]/3):round(3*imcrop.shape[1]/3)]
        imo[n,:,:,:]=imresize(imc1, (dim,dim))
        imo[n+totim,:,:,:]=imresize(imc2, (dim,dim))
        imo[n+(2*totim),:,:,:]=imresize(imc3, (dim,dim))
        imo[n+(3*totim),:,:,:]=imresize(imc4, (dim,dim))
        n=n+1
        print((n/nim)*100)
elif aug == 0:
    while n<totim:
        im=imread(imloc[n])
        imo[n,:,:,:]=im
        n=n+1
        print((n/nim)*100)
    
np.save('img_fr_100', imo)