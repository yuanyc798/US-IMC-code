import warnings
warnings.filterwarnings("ignore")
import cv2
import math
import numpy as np
import sys
import imgaug as ia
from imgaug import augmenters as iaa
import skimage
import skimage.io
import nibabel as nib
import random
import os
import SimpleITK as sitk
from  scipy.stats import ttest_rel
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

colors=[[0,0,0],[0,0,255],[0,255,255],[255,0,0],[0,255,255],[255,255,0]]
def label2color(n_classes, seg):
	seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
	for c in range(1,n_classes):
		seg_color[:, :, 0] += ((seg == c) *(colors[c][0])).astype('uint8')
		seg_color[:, :, 1] += ((seg == c) *(colors[c][1])).astype('uint8')
		seg_color[:, :, 2] += ((seg == c) *(colors[c][2])).astype('uint8')
	seg_color = seg_color.astype(np.uint8)
	return seg_color
def color2label(n_classes, seg):
	seg_color = np.zeros((seg.shape[0], seg.shape[1], 1))
	seg_color1 = np.zeros((seg.shape[0], seg.shape[1], 1))
	seg_color2 = np.zeros((seg.shape[0], seg.shape[1], 1))
	seg_color3 = np.zeros((seg.shape[0], seg.shape[1], 1))
	for c in range(1,n_classes):
		#seg_color[:, :] += ((seg[:, :, 0] == colors[c][0] and seg[:, :,1] == colors[c][1] and seg[:, :, 2] == colors[c][2]) *(c)).astype('uint8')
		seg_color1[:, :,0] = ((seg[:, :,0] == colors[c][0]) *(c)).astype('uint8')
		seg_color2[:, :,0] = ((seg[:, :,1] == colors[c][1]) *(c)).astype('uint8')
		seg_color3[:, :,0] = ((seg[:, :,2] == colors[c][2]) *(c)).astype('uint8')
		seg_color+=seg_color1*seg_color2*seg_color3/(c*c)
	seg_color = seg_color.astype(np.uint8)
	return seg_color
def test_mean_surface_distance():
    """test_mean_surface_distance"""
    x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
    y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
    metric = MeanSurfaceDistance()
    metric.clear()
    metric.update(x, y, 0)
    distance = metric.eval()
    print(distance)

def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
 
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
    return quality

def lregress(ax,data1,data2, *args, **kwargs):
    x=data1
    y=data2
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m1 = 0 
    m2 = 0 
    m3= 0 
    for x_i, y_i in zip(x, y):
        m1 += (x_i - x_mean) * (y_i - y_mean)
        m2 += (x_i - x_mean) ** 2
        m3 += (y_i - y_mean) ** 2
    a = m1/(math.sqrt(m2)*math.sqrt(m3))
    b = y_mean - a*x_mean
    y_line = a*x + b
    print(a,b)
    ax.scatter(data1,data2,marker='.',s=120,c='b',edgecolor='b',*args,** kwargs)
    ax.plot(x, y_line, color='r')
    ax.set_xlabel('Manual',fontsize=14,family='Times New Roman')
    ax.set_ylabel('Proposed',fontsize=14,family='Times New Roman')#round(a, 3)
    ax.text(80, 160, "CC:"+str(round(a, 3)), size = 15,\
         family = "Times New Roman", color = "r", style = "italic", weight = "light",\
         bbox = dict(facecolor = "r", alpha = 0.2))
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
     
def blandaltm(data1,data2, *args, **kwargs):
    fig,(ax2,ax1)= plt.subplots(ncols=2)
    data1=np.asarray(data1.reshape(-1))*(0.07*0.07)
    data2=np.asarray(data2.reshape(-1))*(0.07*0.07)
    mean=np.mean([data1,data2],axis = 0)
    diff=data1-data2
#     print(diff)
    md=np.mean(diff)
    sd=np.std(diff,axis = 0)
    print(diff)
    ax1.text(185,md+1.96 * sd+1.5,'+1.96 SD',fontproperties="Times New Roman", fontsize=16, color = "b", style = "italic",verticalalignment='center', horizontalalignment='right',rotation=0)
    ax1.text(185,md-1.96 * sd+1.5,'-1.96 SD',fontproperties="Times New Roman", fontsize=16, color = "b", style = "italic",verticalalignment='center', horizontalalignment='right',rotation=0)
    ax1.axhline(md,color = 'gray',linestyle = '--',label='md')
    ax1.axhline(md+1.96 * sd,color = 'red',label='md+1.96*sd')
    ax1.axhline(md-1.96 * sd,color = 'red',label='md-1.96*sd')
    print(mean)
    plt.tick_params(labelsize=12)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()

    [label.set_fontname('Times New Roman') for label in labels]
    ax1.scatter(mean,diff,marker='.',s=120,c='b',edgecolor='b',*args,** kwargs)
    
    ax1.set_xlabel('(Manual + Proposed)/2  (mm\u00b2)',fontsize=14,family='Times New Roman')
    ax1.set_ylabel('Manual-Proposed  '+'(mm\u00b2)',fontsize=14,family='Times New Roman')
    lregress(ax2,data2,data1)
    plt.show()     
    
    
dices=0
len1,len2,len3,len4,len5=0,0,0,0,0
dicer,diceg,diceb,diceo,diceq=0,0,0,0,0
ddc=[]
h3=[]
f1=0
Precision=0
Recall=0
nn=np.random.randint(0,3)
print(nn)

pp='./100imt/'
ones=[]

for k in range(100):    
    mask= cv2.imread(pp+str(1*k+1)+'.png',2)//100
    paths='./100fcn db_wloss/'
    pre= cv2.imread(paths+str(k+1)+'.tif', 2)//100#

    ll1=len(mask[mask==1])
    ll2=len(pre[pre==1])
    FN,FP,TP,TN=0   
    
    if len(mask[mask==1])>0:
            len1=len1+1
            mgr=mask*1
            prer=pre*1
            mgr[mgr!=1]=0
            prer[prer!=1]=0
            ma=mgr.astype(np.float32)-prer.astype(np.float32)
            aa=len(ma[ma==0])
            mb=mgr+prer
            bb=len(mb[mb==0])
            
            TN=bb
            TP=aa-bb
            FN=len(ma[ma==1])
            FP=len(ma[ma==-1])             
            recal1=TP/(TP+FN)
            prec1=TP/(TP+FP)
            cc=aa-bb
            dicer+=2*cc/(len(mgr[mgr==1])+len(prer[prer==1]))
            
    if len(mask[mask==2])>0:
            len2=len2+1
            mgr=mask*1
            prer=pre*1
            mgr[mgr!=2]=0
            prer[prer!=2]=0
            mgr=mgr//2
            prer=prer//2            
            ma=mgr.astype(np.float32)-prer.astype(np.float32)
            aa=len(ma[ma==0])
            mb=mgr+prer
            bb=len(mb[mb==0])

            TN+=bb
            TP+=aa-bb
            FN+=len(ma[ma==1])
            FP+=len(ma[ma==-1])             
            recal2=TP/(TP+FN)
            prec2=TP/(TP+FP)
            cc=aa-bb
            diceg+=2*cc/(len(mgr[mgr==1])+len(prer[prer==1]))

    f1+=2*recal2*prec2/(recal2+prec2)
    Precision+=prec2
    Recall+=recal2
           
    ma=mask.astype(np.float32)-pre.astype(np.float32)
    aa = len(ma[ma==0])
    mb=mask+pre
    bb=len(mb[mb==0])
    cc=aa-bb

    m_count = len(mask[mask!=0])
    p_count = len(pre[pre!=0])
    dice=2*cc/(m_count+p_count)
    #print('num '+str(k+1)+':',dice)
    ones.append(ll2)
    ddc.append(ll1)
    dices+=dice
dicem=dices/100

print(' IMT:',diceg/100)
print('lumen:',dicer/100)
print('Recall:',Recall/100)
print('Precision:',Precision/100)
print('F1:',f1/100)

#blandaltm(np.array(ddc),np.array(ones))
#lregress(np.array(ones),np.array(ddc))

