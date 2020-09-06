import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util import random_noise
def gaussian_noisy(image):
    g_noise=random_noise(image, mode='gaussian', seed=None, clip=True)
    return g_noise
  
def sp_noisy(image):
    salt_pepper1=random_noise(image, mode='s&p', seed=None, clip=True)
    return salt_pepper1

def speckle_noisy(image):
    s_noisy=random_noise(image, mode='speckle', seed=None, clip=True)
    return s_noisy
  
def border_noisy(image):
    imgSize = 28
    bor_noisy = np.ones(image.shape)
    bor_noisy[2:imgSize-2,2:imgSize-2] = image[2:imgSize-2,2:imgSize-2]
    return bor_noisy
  
def block_noisy(image):
    imgSize = 28
    c = np.random.randint(0,int(imgSize/2));
    r = np.random.randint(0,int(imgSize/2));
    locc = np.random.randint(0,imgSize-int(c));
    locr = np.random.randint(0,imgSize-int(r));
    blk_noisy = image.copy();
    blk_noisy[locr:locr+r,locc:locc + c,:] = 1;
    return blk_noisy
     

def load_MNIST():
    data_train = pd.read_csv('/home/guest/Desktop/spyder-project/playwithMNIST/MNIST/mnist_train.csv').values
    X_train , y_train = data_train[:,1:], data_train[:,:1]

    data_test = pd.read_csv('/home/guest/Desktop/spyder-project/playwithMNIST/MNIST/mnist_test.csv').values
    X_test , y_test = data_test[:,1:], data_test[:,:1]
    
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    return ( X_train , y_train),(X_test , y_test)

def load_gaussian(images):
    gaussian_n_data_train = []
    for i in images:
        image = np.reshape(i,(28,28,1))
        n_image = gaussian_noisy(image)
#        n_image = np.reshape(n_image,(784,))
        gaussian_n_data_train.append(n_image)
    return np.array(gaussian_n_data_train)

def load_sp(images):
   sp_n_data_train = []
   for i in images:
       image = np.reshape(i,(28,28,1))
       n_image = sp_noisy(image)
#       n_image = np.reshape(n_image,(784,))
       sp_n_data_train.append(n_image)
   return np.array(sp_n_data_train)

def load_speckle(images):
    speckle_n_data_train = []
    for i in images:
        image = np.reshape(i,(28,28,1))
        n_image = speckle_noisy(image)
#        n_image = np.reshape(n_image,(784,))
        speckle_n_data_train.append(n_image)
    return np.array(speckle_n_data_train)


def load_border(images):
    border_n_data_train = []
    for i in images:
        image = np.reshape(i,(28,28,1))
        n_image = border_noisy(image)
#        n_image = np.reshape(n_image,(784,))
        border_n_data_train.append(n_image)
    return np.array(border_n_data_train)


def load_block(images):
    block_n_data_train = []
    for i in images:
        image = np.reshape(i,(28,28,1))
        n_image = block_noisy(image)
#        n_image = np.reshape(n_image,(784,))
        block_n_data_train.append(n_image)
    return np.array(block_n_data_train)



#plt.imshow(np.reshape(X_train[0,:],(28,28)))
#p = sp_noisy(np.reshape(X_train[0,:],(28,28,1)))
#plt.imshow(p[:,:,0])
#print(psnr(X_train[0,:],X_train[0,:]))




    


    