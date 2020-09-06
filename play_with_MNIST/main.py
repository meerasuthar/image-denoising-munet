# -*- coding: utf-8 -*-

from playwithMNIST.datamaker import *
from playwithMNIST.helper import PSNR, plot_image
from playwithMNIST.modelmaker import unet_v1

import matplotlib.pyplot as plt
import numpy as np


(clean_train, y_train), (clean_test, y_test) =  load_MNIST()
(sp_train,sp_test) = load_sp(clean_train), load_sp(clean_test)
(block_train,block_test) = load_block(clean_train), load_block(clean_test)
(speckle_train,speckle_test) = load_speckle(clean_train), load_speckle(clean_test)
(border_train,border_test) = load_border(clean_train), load_border(clean_test)
(gaussian_train,gaussian_test) = load_gaussian(clean_train), load_gaussian(clean_test)

clean_train = clean_train.reshape(clean_train.shape[0],28,28,1)
clean_test = clean_test.reshape(clean_test.shape[0],28,28,1)

#plot_image(clean_train[0])
#plot_image(sp_train[0])
#plot_image(block_train[0])
#plot_image(speckle_train[0])
#plot_image(border_train[0])
#plot_image(gaussian_train[0])

print(PSNR(clean_train,sp_train))
print(PSNR(clean_train,block_train))
print(PSNR(clean_train,speckle_train))
print(PSNR(clean_train,border_train))
print(PSNR(clean_train,gaussian_train))

autoencoder_border, encoder_border = unet_v1()
autoencoder_border.fit(border_train,
            clean_train,
            epochs = 20,
            batch_size=128,
            verbose=2)
border_pred = autoencoder_border.predict(border_test)
border_pred_train = autoencoder_border.predict(border_train)
border_latent = encoder_border.predict(border_test)
border_latent_train = encoder_border.predict(border_train)


autoencoder_gaussian, encoder_gaussian = unet_v1()
autoencoder_gaussian.fit(gaussian_train,
            clean_train,
            epochs = 20,
            batch_size=128,
            verbose=2)
gaussian_pred = autoencoder_gaussian.predict(gaussian_test)
gaussian_pred_train = autoencoder_gaussian.predict(gaussian_train)
gaussian_latent = encoder_gaussian.predict(gaussian_test)
gaussian_latent_train = encoder_gaussian.predict(gaussian_train)


autoencoder_sp, encoder_sp = unet_v1()
autoencoder_sp.fit(sp_train,
            clean_train,
            epochs = 20,
            batch_size=128,
            verbose=2)
sp_pred = autoencoder_sp.predict(sp_test)
sp_pred_train = autoencoder_sp.predict(sp_train)
sp_latent = encoder_sp.predict(sp_test)
sp_latent_train = encoder_sp.predict(sp_train)

print('Border UNet', PSNR(border_pred,clean_test))
print('Gaussian UNet', PSNR(gaussian_pred,clean_test))
print('S&P UNet', PSNR(sp_pred,clean_test))


autoencoder_border.trainable = False
autoencoder_gaussian.trainable = False
autoencoder_sp.trainable = False

encoder_border.trainable = False
encoder_gaussian.trainable = False
encoder_sp.trainable = False

# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,concatenate,Flatten,Dropout,Multiply,Add
from keras.models import Model
from keras import backend as K
    


valve_train = np.concatenate((border_train, gaussian_train, sp_train), axis=0)
valve_train_out = np.concatenate((clean_train, clean_train, clean_train), axis=0)

Valve.fit(valve_train, valve_train_out, epochs=50, batch_size=128, verbose=2)

valve_border = Valve.predict(border_test)
valve_gaussian = Valve.predict(gaussian_test)
valve_sp = Valve.predict(sp_test)

print('Border Valve', PSNR(valve_border,clean_test))
print('Gaussian Valve', PSNR(valve_gaussian,clean_test))
print('S&P Valve', PSNR(valve_sp,clean_test))

print('Border UNet', PSNR(autoencoder_border.predict(border_test),clean_test))
print('Border UNet Gaussian Noise', PSNR(autoencoder_border.predict(gaussian_test),clean_test))
print('Border UNet S&P Noise', PSNR(autoencoder_border.predict(sp_test),clean_test))

print('gaussian UNet', PSNR(autoencoder_gaussian.predict(border_test),clean_test))
print('gaussian UNet Gaussian Noise', PSNR(autoencoder_gaussian.predict(gaussian_test),clean_test))
print('gaussian UNet S&P Noise', PSNR(autoencoder_gaussian.predict(sp_test),clean_test))

print('sp UNet', PSNR(autoencoder_sp.predict(border_test),clean_test))
print('sp UNet Gaussian Noise', PSNR(autoencoder_sp.predict(gaussian_test),clean_test))
print('sp UNet S&P Noise', PSNR(autoencoder_sp.predict(sp_test),clean_test))

print('Gaussian UNet', PSNR(gaussian_pred,clean_test))
print('S&P UNet', PSNR(sp_pred,clean_test))

autoencoder_border.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/autoencoder_border.h5')
autoencoder_gaussian.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/autoencoder_gaussian.h5')
autoencoder_sp.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/autoencoder_sp.h5')

encoder_border.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/encoder_border.h5')
encoder_gaussian.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/encoder_gaussian.h5')
encoder_sp.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/encoder_sp.h5')

Valve.save_weights('/home/guest/Desktop/spyder-project/playwithMNIST/saved_models/Valve.h5')

####################### MULTIPLE NOISES

#noisy_test = (border_test + gaussian_test + sp_test) / 3

noisy_test = load_border(load_gaussian(load_sp(clean_test)))

m1 = autoencoder_border.predict(noisy_test)
m2 = autoencoder_gaussian.predict(noisy_test)
m3 = autoencoder_sp.predict(noisy_test)
valve_pred = Valve.predict(noisy_test)

print(PSNR(m1+m2+m3-2*noisy_test, clean_test))
print(PSNR(m1, clean_test))
print(PSNR(m2, clean_test))
print(PSNR(m3, clean_test))
print(PSNR(valve_pred, clean_test))

