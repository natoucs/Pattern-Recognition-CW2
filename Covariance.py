import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.spatial.distance as sc_dist
from PIL import Image
from scipy.misc import imshow
from scipy.misc import toimage
import time
import math
from sklearn import preprocessing

my_data = genfromtxt('wine.data.csv', delimiter='')


train=[]
test=[]
for i in range (178):
    if(my_data[i,0]==1):
        train.append(my_data[i,1:])
    else:         
        test.append(my_data[i,1:])

train=np.array(train).T
test=np.array(test).T

train[1:,:]=preprocessing.scale(train[1:,:],with_mean=False)
test[1:,:]=preprocessing.scale(test[1:,:],with_mean=False)


average_train=np.array([np.mean(train[1:,:],axis=1)]*118).T
Atrain=train[1:,:]-average_train
covariance_st=Atrain.dot(Atrain.T)

train=[]
test=[]
for i in range (178):
    if(my_data[i,0]==1):
        train.append(my_data[i,1:])
    else:         
        test.append(my_data[i,1:])

train=np.array(train).T
test=np.array(test).T


train[1:,:]=train[1:,:]/np.linalg.norm(train[1:,:],axis=0)
test[1:,:]=test[1:,:]/np.linalg.norm(test[1:,:],axis=0)

average_train=np.array([np.mean(train[1:,:],axis=1)]*118).T
Atrain=train[1:,:]-average_train
covariance_norm=Atrain.dot(Atrain.T)

train=[]
test=[]
for i in range (178):
    if(my_data[i,0]==1):
        train.append(my_data[i,1:])
    else:         
        test.append(my_data[i,1:])

train=np.array(train).T
test=np.array(test).T

average_train=np.array([np.mean(train[1:,:],axis=1)]*118).T
Atrain=train[1:,:]-average_train
covariance_ori=Atrain.dot(Atrain.T)

print(covariance_ori/covariance_st)