# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,concatenate
from keras.models import Model
from keras import backend as K
    

def unet_v1(input_shape = (28,28,1)):
   
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    
    e1 = Conv2D(16, (3, 3), activation='relu',padding = 'same')(input_img)
    p1 = MaxPooling2D((2, 2))(e1)
    e2 = Conv2D(8, (3, 3), activation='relu',padding = 'same')(p1)
    encoded = MaxPooling2D((2, 2))(e2)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    
    d3 = Conv2D(8, (3, 3), activation='relu',padding = 'same')(encoded)
    u1 = UpSampling2D((2, 2))(d3)
    con = concatenate([u1,e2],axis = 3)
    d2 = Conv2D(16, (3, 3), activation='relu',padding = 'same')(con)
    u2 = UpSampling2D((2, 2))(d2)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u2)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    encoder = Model(input_img, encoded)
    encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    autoencoder.summary()
    return autoencoder, encoder
    