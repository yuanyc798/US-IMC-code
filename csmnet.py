import numpy as np
import tensorflow as tf
import pickle
from random import shuffle
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
import cv2
import os
import sys
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import skimage
import skimage.io
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy
#from tensorflow.keras.applications.resnet50 import ResNeXt50

print("TensorFlow Version: {}".format(tf.__version__))

def senet(x):
    num=int(x.shape[3])
    squeeze =tf.reduce_mean(x, axis=[1,2])
    excitation = tf.layers.dense(inputs=squeeze,units=num//16)
    excitation = tf.nn.relu(excitation)
    excitation = tf.layers.dense(inputs=excitation,units=num)
    excitation = tf.nn.sigmoid(excitation)
    excitation = tf.reshape(excitation,[-1,1, 1, num])
    scale = x*excitation
    return scale 

def conBlock(img,channel,is_train):
        layer1 = tf.layers.conv2d(img,channel, 3, strides=1, padding='same')
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        layer11 = tf.nn.relu(layer1)#tf.maximum(alpha * layer1, layer1)
        
        layer = tf.layers.conv2d(layer11,channel, 3, strides=1, padding='same')
        layer1 = tf.layers.batch_normalization(layer, training=is_train)
        layer2 = tf.nn.relu(layer1)#tf.maximum(alpha * layer1, layer1)
        return layer2        

def tsam(x):
    num=int(x.shape[3])
    sm =tf.reduce_mean(x, axis=-1)
    sn =tf.reduce_max(x, axis=-1)    
    sm=tf.expand_dims(sm,-1)   
    sn=tf.expand_dims(sn,-1)
    con1=tf.layers.conv2d(x,1,1,strides=1,padding='same')
    shj=tf.concat([sm,sn,con1], axis=-1) 
            
    layer1 = tf.layers.conv2d(shj,1,3,strides=1, padding='same')
    excitation = tf.nn.sigmoid(layer1)

    scale = x*excitation
    return scale   
           
                		
def cdcfr(x,numb,is_train):
        numb=int(numb/2)
        layer1 = tf.layers.conv2d(x,numb,3, strides=1, padding='same')
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        layer11 = tf.nn.relu(layer1)
		
        layer2 = tf.layers.conv2d(x,numb,3, strides=1, padding='same',dilation_rate=(4,4))
        layer21 = tf.layers.batch_normalization(layer2, training=is_train)
        layer21 = tf.nn.relu(layer21)

        layer3 = tf.layers.conv2d(x,numb,3, strides=1, padding='same',dilation_rate=(2,2))
        layer31 = tf.layers.batch_normalization(layer3, training=is_train)
        layer31 = tf.nn.relu(layer31)

        layer4 = tf.layers.conv2d(x,numb,3, strides=1, padding='same',dilation_rate=(3,3))
        layer41 = tf.layers.batch_normalization(layer4, training=is_train)
        layer41 = tf.nn.relu(layer41)
        
        #conf = ccdia3(x,numb,is_train)
        con= tf.concat([layer11,layer21,layer31,layer41],axis=-1)#12
        
        layer2 = tf.layers.conv2d(con,numb,3, strides=1, padding='same',dilation_rate=(4,4))
        layer21 = tf.layers.batch_normalization(layer2, training=is_train)
        layer21 = tf.nn.relu(layer21)
        #layer21=tf.add(layer11,layer21)

        layer3 = tf.layers.conv2d(con,numb,3, strides=1, padding='same',dilation_rate=(2,2))
        layer31 = tf.layers.batch_normalization(layer3, training=is_train)
        layer31 = tf.nn.relu(layer31)
        #layer31=tf.add(layer11,layer31)
        
        layer4 = tf.layers.conv2d(con,numb,3, strides=1, padding='same',dilation_rate=(3,3))
        layer41 = tf.layers.batch_normalization(layer4, training=is_train)
        layer41 = tf.nn.relu(layer41)
        #layer41=tf.add(layer11,layer41)
        
        con= tf.concat([layer11,layer21,layer31,layer41],axis=-1)
        con=senet(con)        
        return con

def upRsize(x,img_h,img_w):
		return tf.image.resize_images(x,(img_h,img_w),method=0)

def csmnet(img,name, output_dim, is_train, alpha=0.01):#288 512
	with tf.variable_scope(name, reuse=(not is_train)):
		con1=conBlock(img,16,is_train)

		h=img.shape[1]
		w=img.shape[2]
		print(con1.shape)
		pool1=tf.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2])(con1)#tf.layers.max_pooling2d
		mg1=tf.image.resize_images(img,(h//2,w//2),method=0)
		mg1c=conBlock(mg1,16,is_train)
		mg1c=tf.concat([mg1c,pool1], axis=-1)
		con2=conBlock(mg1c,32,is_train)

        		
		pool2=tf.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2])(con2)
		mg2=tf.image.resize_images(img,(h//4,w//4),method=0)
		mg2c=conBlock(mg2,32,is_train)
		mg2c=tf.concat([mg2c,pool2], axis=-1)		
		con3=conBlock(mg2c,64,is_train)
		
		pool3=tf.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2])(con3)
		mg3=tf.image.resize_images(img,(h//8,w//8),method=0)
		mg3c=conBlock(mg3,64,is_train)
		mg3c=tf.concat([mg3c,pool3], axis=-1)		
		con4=conBlock(mg3c,80,is_train)

		pool4=tf.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2])(con4)
		mg4=tf.image.resize_images(img,(h//16,w//16),method=0)
		mg4c=conBlock(mg4,80,is_train)
		mg4c=tf.concat([mg4c,pool4], axis=-1)		
		con5=conBlock(mg4c,96,is_train)


		pool5=tf.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2])(con5)
		mg5=tf.image.resize_images(img,(h//32,w//32),method=0)
		mg5c=conBlock(mg5,128,is_train)
		mg5c=tf.concat([mg5c,pool5], axis=-1)		
		con=conBlock(mg5c,128,is_train)#
        
		con=cdcfr(con,128,is_train)
   
		x0= tf.layers.conv2d(con,output_dim,1, strides=1, padding='same')
		outputs0=tf.nn.softmax(x0)   
        
		layer2 =tf.image.resize_images(con,(h//16,w//16),method=0) 
		conc1=tf.concat([(con5),layer2], axis=-1)   
		conde1=conBlock(conc1,96,is_train)
		conde1=tsam(conde1)
		x1=tf.image.resize_images(conde1,(h,w),method=0) 
		x1= tf.layers.conv2d(conde1,output_dim,1, strides=1, padding='same')
		outputs1=tf.nn.softmax(x1)		

		layer3 =tf.image.resize_images(conde1,(h//8,w//8),method=0)
		conc2=tf.concat([(con4),layer3], axis=-1) 
		conde2=conBlock(conc2,80,is_train)
		conde2=tsam(conde2)
		x2=tf.image.resize_images(conde2,(h,w),method=0)
		x2= tf.layers.conv2d(conde2,output_dim,1, strides=1, padding='same')
		outputs2=tf.nn.softmax(x2)
		
		layer4 =tf.image.resize_images(conde2,(h//4,w//4),method=0)
		conc3=tf.concat([(con3),layer4], axis=-1) 
		conde3=conBlock(conc3,48,is_train)
		conde3=tsam(conde3)
		x3=tf.image.resize_images(conde3,(h,w),method=0)
		x3= tf.layers.conv2d(conde3,output_dim,1, strides=1, padding='same')
		outputs3=tf.nn.softmax(x3)

		layer5 =tf.image.resize_images(conde3,(h//2,w//2),method=0)
		conc4=tf.concat([(con2),layer5], axis=-1)   
		conde4=conBlock(conc4,32,is_train)
		conde4=tsam(conde4)
		x4=tf.image.resize_images(conde4,(h,w),method=0)
		x4= tf.layers.conv2d(conde4,output_dim,1, strides=1, padding='same')
		outputs4=tf.nn.softmax(x4)
		
		layer6 =tf.image.resize_images(conde4,(h,w),method=0) 
		conc5=tf.concat([(con1),layer6], axis=-1) 
		conde5=conBlock(conc5,32,is_train)
		conde5=tsam(conde5)	
		x5=tf.image.resize_images(conde5,(h,w),method=0)	
		x= tf.layers.conv2d(x5,output_dim,1, strides=1, padding='same')
		outputs=tf.nn.softmax(x)#outputs = tf.tanh(logits)  tf.nn.softmax(x)
		
		return outputs1,outputs2,outputs3,outputs4,outputs	

   
