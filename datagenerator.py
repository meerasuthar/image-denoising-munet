# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

def make_patchs(noise,clean,size=16, dataset = 10000):
    noise = scipy.io.loadmat(noise)['IMAGES']
    clean = scipy.io.loadmat(clean)['IMAGES']
    
    X = []
    Y = []
    for i in range(noise.shape[2]):
        single_noise = noise[:,:,i:i+1]
        single_clean = clean[:,:,i:i+1]
        for _i in range(dataset):
            x = np.random.randint(512-size)
            y = np.random.randint(512-size)
            X.append(single_noise[x:x+size,y:y+size,:])
            Y.append(single_clean[x:x+size,y:y+size,:])
    return np.array(X), np.array(Y)
        

#path = '/home/guest/Desktop/spyder-project/playwithMNIST/brain/BrainImagesAll/brainTest/'
#noise = path+'Gt4IMAGES.mat'
#clean = path+'IMAGES.mat'
#X,Y = make_patchs(noise, clean)
#
#from modelmaker import *
#autoencoder, encoder = unet_v1()
#autoencoder.fit(X,
#            Y,
#            epochs = 20,
#            batch_size=128,
#            verbose=2)
#
#sp_pred_train = autoencoder.predict(X)
#sp_latent_train = encoder.predict(Y)
#
#
#from helper import PSNR
#
#print(PSNR(sp_pred_train,Y))
#
#tmp = np.array([scipy.io.loadmat(noise)['IMAGES'][:,:,0:1]])
#pred = autoencoder.predict(tmp)
#plt.imshow(pred[0,:,:,0])
#
#
#plt.imshow(scipy.io.loadmat(clean)['IMAGES'][:,:,0])
#
#plt.imshow(scipy.io.loadmat(noise)['IMAGES'][:,:,0])
#
#print(PSNR(pred[0,:,:,0],scipy.io.loadmat(clean)['IMAGES'][:,:,0]))