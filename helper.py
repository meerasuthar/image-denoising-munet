# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def PSNR(y1,y2):
    m = ((y1-y2)**2).mean()
    return -10*np.log10(m)

def plot_image(x):
    plt.imshow(np.reshape(x,(28,28)), cmap=cm.gray)
    
    
# -*- coding: utf-8 -*-

