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
from csmnet import *

def dice_n(target,output):
    smooth=1e-10
    target=np.argmax(target,3)
    target.astype(np.float32)
    output=np.argmax(output,3)
    output.astype(np.float32)
    
    inse=np.sum(target*output,axis=(1,2))
    l=np.sum(output*output,axis=(1,2))
    r=np.sum(target*target,axis=(1,2))
    
    dice=(2. * inse + smooth) / (l + r + smooth)
    dice=np.mean(dice)
    return dice
    #return 0.6*d_loss+0.4*tf.reduce_mean(categorical_crossentropy(y_true, y_pred))

def bin_csnpy(y_true,y_pred):
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	return tf.reduce_mean(binary_crossentropy(y_true, y_pred))
def dice_coe(y_true,output,axis=[1,2], smooth=1e-10):
    inse = tf.reduce_sum(output * y_true, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(y_true * y_true, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice 
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    d_loss = 1-dice_coe(y_true, y_pred)  
    return d_loss  
def dice_ls(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    d_loss = tf.reduce_sum(0.33*(1-dice_coe(y_true[:,:,:,0], y_pred[:,:,:,0]))+\
                           0.33*(1-dice_coe(y_true[:,:,:,1], y_pred[:,:,:,1]))+0.33*(1-dice_coe(y_true[:,:,:,2], y_pred[:,:,:,2])))                                  
    return 1*d_loss     
            
def dc_los_bcos(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)   
    d_loss =(0.25*(1-dice_coe(y_true[:,:,:,0],y_pred[:,:,:,0]))+0.25*(1-dice_coe(y_true[:,:,:,1],y_pred[:,:,:,1]))+0.5*(1-dice_coe(y_true[:,:,:,2],y_pred[:,:,:,2]))) 
    return 0.6*d_loss+0.4*(0.25*bin_csnpy(y_true[:,:,:,0],y_pred[:,:,:,0])+0.25*bin_csnpy(y_true[:,:,:,1],y_pred[:,:,:,1])+0.5*bin_csnpy(y_true[:,:,:,2],y_pred[:,:,:,2])) 


def upRsize(x,img_h,img_w):
		
		return tf.image.resize_images(x,(img_h,img_w),method=0)
def network( inputs_img1,output_dim, is_train, alpha=0.01):
    outputs1,outputs2,outputs3,outputs4,outputs=csmnet(inputs_img1,'fcn1', output_dim, is_train)
    return outputs1,outputs2,outputs3,outputs4,outputs
        
def get_loss(inputs_lab1,inputs_img1,output_dim, smooth=0.1):
    h=inputs_img1.shape[1]
    w=inputs_img1.shape[2]
    #outputs0,output1,output2,outputs3,outputs4,outputs= network2(inputs_img1,output_dim, is_train=True)
    output1,output2,outputs3,outputs4,outputs= network(inputs_img1,output_dim, is_train=True)    
    lab4=tf.image.resize_images(inputs_lab1,(h//2,w//2),method=0)
    lab3=tf.image.resize_images(inputs_lab1,(h//4,w//4),method=0)
    lab2=tf.image.resize_images(inputs_lab1,(h//8,w//8),method=0)
    lab1=tf.image.resize_images(inputs_lab1,(h//16,w//16),method=0)
    #lb5=tf.image.resize_images(inputs_lab1,(h//32,w//32),method=0)    
    
    #loss0 = dc_los_bcos(lb5,outputs0)    
    loss1 = dc_los_bcos(lab1,output1)
    loss2 = dc_los_bcos(lab2,output2)
    loss3 = dc_los_bcos(lab3,outputs3)
    loss4 = dc_los_bcos(lab4,outputs4)
    loss = dc_los_bcos(inputs_lab1,outputs)    
    return 0.2*loss1+0.2*loss2+0.2*loss3+0.2*loss4+0.2*loss
    #return 1*loss
    
    
def show_generator_output(sess,Va_images,inputs_img1,output_dim):
    cmap = 'Greys_r'
    sp1,sp2,sp3,sp4,sp5= sess.run(network(inputs_img1,output_dim, False),feed_dict={inputs_img1: Va_images})               
    return sp5
    

     
        
