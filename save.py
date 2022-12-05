import numpy as np
import tensorflow as tf
from random import shuffle
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
import cv2
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy
import time
from loss import *

def save_images(samples,id_test,path_save,Yv_hot):
    roctrix=np.zeros((4,30))
    isExists=os.path.exists(path_save)
    nm=10
    if not isExists:
        os.makedirs(path_save)
    for i in range(nm):
      cv2.imwrite(path_save+id_test[i].split('/')[-1].split('.')[0]+'.tif',np.argmax(samples[i],-1)*100)
      preds=samples[i]
      
      for k in range(1,3):#(1,3)
       Yv_hot= np.array(Yv_hot)
       mlab=Yv_hot[i,:,:,k]
       for m in range(30):
        roc=preds[:,:,k]*1.0                         
        roc[roc>(0.1*m)]=1
        roc[roc<(0.1*m)]=0

        roc=roc.astype(np.float32)
        TP0=mlab-roc
        TP=len(TP0[TP0==0])
      
        TN0=mlab+roc
        TN=len(TN0[TN0==0])   
   
        FP0=mlab-roc
        FP=len(FP0[FP0==-1])      

        FN0=mlab-roc
        FN=len(FN0[FN0==1])          
        roctrix[0,m]+=TP
        roctrix[1,m]+=TN
        roctrix[2,m]+=FP
        roctrix[3,m]+=FN
    return roctrix
    

    
     
        
