import numpy as np
import tensorflow as tf
import pickle
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
import skimage
import skimage.io
from csmnet import *
from loss import *
from save import *
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("TensorFlow Version: {}".format(tf.__version__))
import time

def get_seq():
     sometimes = lambda aug: iaa.Sometimes(0.3, aug)
     seq = iaa.Sequential([iaa.Fliplr(0.4),iaa.Flipud(0.3),sometimes(iaa.Crop()),
                           sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},rotate=(-45, 45),shear=(-15, 15),cval=0))
                           ],random_order=True)
     return seq 

def get_inputs(img_dim, image_height, image_width, image_depth):
    inputs_lab1 = tf.placeholder(tf.float32, [None, image_height, image_width, img_dim], name='inputs_lab')
    inputs_img1 = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_img')
    return inputs_lab1,inputs_img1#,inputs_img2

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

def upRsize(x,img_h,img_w):
		return tf.image.resize_images(x,(img_h,img_w),method=0)

def network( inputs_img1,output_dim, is_train, alpha=0.01):
    outputs1,outputs2,outputs3,outputs4,outputs=csmnet(inputs_img1,'fcn1', output_dim, is_train)
    return outputs1,outputs2,outputs3,outputs4,outputs
   
def get_optimizer(g_loss,beta1=0.9, learning_rate=0.001):
    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars ]#if var.name.startswith("generator")
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    #g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss)
    return g_opt

def lab_path(file_path):
    return  file_path.split('.')[0]+'.png'
def batchsize(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size)) 
def preprocess_input(x):#BGR
    #x = skimage.color.rgb2gray(x) 
    x = (x - np.mean(x))/255
    return x
def augmentImg(seq,X,Y,augment):
		if augment:
			seq_det = seq.to_deterministic()
			X_aug = [seq_det.augment_image(x) for x in X]                
			Y_map1 = [ia.SegmentationMapOnImage(np.squeeze(y).astype(np.uint8), shape=(320,512,1), nb_classes=3) for y in Y]
			Y_aug1 = [seq_det.augment_segmentation_maps([y])[0].get_arr_int().astype(np.uint8) for y in Y_map1]
			X = X_aug
			Y= Y_aug1
		return X,Y
        
def ModelVal(sess,saver,id_img_val,inputs_img1,lab_s,dicei,path):
					#global dicei
					V = [((skimage.io.imread(x))) for x in id_img_val]
					Va_images =[preprocess_input(x) for x in V]
					Va_images = np.array(Va_images)
					
					Yv=[np.expand_dims(cv2.imread(lab_path(x),2)/100,-1) for x in id_img_val]
					Yv_hot = [to_categorical(y,lab_s) for y in Yv]
					samples = show_generator_output(sess,Va_images,inputs_img1,lab_s)
					
					dicev=dice_n(Yv_hot,samples)
					if dicev>dicei:
						save_path = saver.save(sess,path)
						dicei=dicev
						print('dice changed,model saved---------  ','val dice:',dicei)            
					else:
						print('val_dice:',dicev,'max dice:',dicei)
					return dicei                                                       
def Modeltst(sess,saver,id_img_test,inputs_img1,lab_s,fold_test,path):
				#global path
				saver.restore(sess,path)
				V = [((skimage.io.imread(x))) for x in id_img_test]
				tst_images =[preprocess_input(x) for x in V]
				tst_images = np.array(tst_images)
				Yv=[np.expand_dims(cv2.imread(lab_path(x),2)/100,-1) for x in id_img_test]
				Yv_hot = [to_categorical(y,lab_s) for y in Yv]
                
				samples = show_generator_output(sess,tst_images,inputs_img1,lab_s)
				matrix=save_images(samples,fold_test,path_save,Yv_hot)
				return matrix
def ROC():
    rmatrix=np.zeros((2,30))
    for t in range(30):
            TP=rocmatrix[0,t]
            TN=rocmatrix[1,t]
            FP=rocmatrix[2,t]
            FN=rocmatrix[3,t]
            TPR=TP/(TP+FN)
            FPR=FP/(FP+TN)
            rmatrix[0,t]=FPR
            rmatrix[1,t]=TPR
            
    f = open(path_save+"ROC.txt",'a')
    for i in range(2):
      for j in range(30):
         f.write(str(rmatrix[i,j]))
         f.write(',')     
      f.write('\n')        

trainimage='./100imt/'
nt=10
ns=10
path_save = r'./100cfm test/'
#global rocmatrix
rocmatrix=np.zeros((4,30))
for it in range(0,nt):
		fold_test=it+1
		id_img_train=[]
		id_img_test=[]
		id_img_val=[]   
		for i in range(0,ns*it):
			id_img_train.append(trainimage+str(i+1)+r'.jpg')

		id_img_train.append(trainimage+str(ns*(it+1)+9)+r'.jpg')
		for i in range(ns*(it+2),100):
			id_img_train.append(trainimage+str(i+1)+r'.jpg')
            
		if it==9:
			id_img_train=[]
			for i in range(9,ns*it):
				id_img_train.append(trainimage+str(i+1)+r'.jpg')   
			for i in range(0,9):
				id_img_val.append(trainimage+str(i+1)+r'.jpg')
		if it<9:
			for i in range(0,9):
				id_img_val.append(trainimage+str(ns*(it+1)+i+1)+r'.jpg')      
		for i in range(0,ns):
			id_img_test.append(trainimage+str(ns*it+i+1)+r'.jpg')
		print('train:',len(id_img_train))
		print('val:',len(id_img_val))
		print('tst:',len(id_img_test))
		batch_size =3
		epochs =100
		learning_rate = 0.001
		beta1 = 0.9
		print('----------image  model:',fold_test)
		def train(data_shape, batch_size,augment):
			global rocmatrix   
			losses = []
			dicei=0
			inputs_lab1, inputs_img1= get_inputs(data_shape[0], data_shape[1], data_shape[2], data_shape[3])
			lab_s=data_shape[0]
			g_loss= get_loss(inputs_lab1,inputs_img1,lab_s)
			g_train_opt= get_optimizer(g_loss,beta1, learning_rate)
			tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				saver = tf.train.Saver()
				seq = get_seq()
				path='./result/tmodels'+str(fold_test)+'.ckpt'#tmodelx2
				#epochs=100
				for e in range(epochs):
					shuffle(id_img_train)
					train_loss_g=0
					for batch in batchsize(id_img_train, batch_size):
						X = [(skimage.io.imread(x)) for x in batch] 
						Y = [np.expand_dims(cv2.imread(lab_path(x),2)/100,-1) for x in batch]

						X,Y=augmentImg(seq,X,Y,augment)
						batch_images =[preprocess_input(x) for x in X]
						batch_images = np.array(batch_images)
						
						Y_hot = [to_categorical(y,lab_s) for y in Y]#print(batch_images.shape)
						_ = sess.run(g_train_opt, feed_dict={inputs_lab1: Y_hot,inputs_img1: batch_images})
						train_loss_g= train_loss_g + g_loss.eval({inputs_lab1: Y_hot,inputs_img1: batch_images})
					print("Epoch {}/{},".format(e+1, epochs),
						  " Loss: {:.4f},". format(train_loss_g*(batch_size)/len(id_img_train)))
					dicei=ModelVal(sess,saver,id_img_val,inputs_img1,lab_s,dicei,path)#validation

				rocmatrix+=Modeltst(sess,saver,id_img_test,inputs_img1,lab_s,fold_test,path)
                
		with tf.Graph().as_default():
			train([3, 320, 512,3], batch_size,True)
ROC()

