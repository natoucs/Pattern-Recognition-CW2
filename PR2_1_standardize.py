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

results={}


predicted=[]
for i in range(60): #compute the the prediction for each point of test set
    distance_manhattan=[] #list that remembers the distance between every train point and the current test point
    for j in range(118): #compute distance of the test point with all the train point
        manhattan=sc_dist.cityblock(test[1:,i],train[1:,j]) #actual distance (1: to discard the first dimension in the vector which is the class, we don't want it in the computation of the distance) - in general 1: takes all the value after index1 in the array, opposite of :1
        distance_manhattan.append(manhattan) #add the distance to the array
    
    distance_manhattan=np.array(distance_manhattan) #transform list in array
    predicted.append(train[0,np.argmin(distance_manhattan)]) #add the class of the vector which had the minimum distance  with this specific test point to the predicted vector

classification_error= np.count_nonzero(predicted==test[0,:]) #count the number of correctly predicted points , count non_zero will count the True Value from the logical array created by the condition
        
results["manhattan"]=(60-classification_error)/60 #add the results in a dictionnary, 60-X/60 is to have the percentage of incorrectly classified points (whereas classification_error gives the number of correctly classified points)

predicted=[]
for i in range(60):
    distance_euclidean=[]
    for j in range(118):
        euclidean=sc_dist.euclidean(test[1:,i],train[1:,j])
        distance_euclidean.append(euclidean)
    
    distance_euclidean=np.array(distance_euclidean)
    predicted.append(train[0,np.argmin(distance_euclidean)])

classification_error= np.count_nonzero(predicted==test[0,:])

results["euclidean"]=(60-classification_error)/60


average_train=np.array([np.mean(train[1:,:],axis=1)]*118).T
Atrain=train[1:,:]-average_train
covariance=Atrain.dot(Atrain.T)
inverse=np.linalg.inv(covariance)


# average_train=np.array([np.mean((my_data[:,2:].T),axis=1)]*178).T
# Atrain=(my_data[:,2:].T)-average_train
# covariance=Atrain.dot(Atrain.T)
# inverse=np.linalg.inv(covariance)
# print(inverse.shape)


predicted=[]
for i in range(60):
    distance_mahal=[]
    for j in range(118):
        mahal=sc_dist.mahalanobis(test[1:,i],train[1:,j],inverse)
        distance_mahal.append(mahal)
    
    distance_mahal=np.array(distance_mahal)
    predicted.append(train[0,np.argmin(distance_mahal)])

classification_error= np.count_nonzero(predicted==test[0,:])

results["mahalanobis"]=(60-classification_error)/60


predicted=[]
for i in range(60):
    distance_cosine=[]
    for j in range(118):
        cosine=sc_dist.cosine(test[1:,i],train[1:,j])
        distance_cosine.append(cosine)
    
    distance_cosine=np.array(distance_cosine)
    predicted.append(train[0,np.argmin(distance_cosine)])

classification_error= np.count_nonzero(predicted==test[0,:])

results["cosine"]=(60-classification_error)/60


predicted=[]
for i in range(60):
    distance_correlation=[]
    for j in range(118):
        correlation=sc_dist.correlation(test[1:,i],train[1:,j])
        distance_correlation.append(correlation)
    
    distance_correlation=np.array(distance_correlation)
    predicted.append(train[0,np.argmin(distance_correlation)])

classification_error= np.count_nonzero(predicted==test[0,:])

results["correlation"]=(60-classification_error)/60


predicted=[]
for i in range(60):
    distance_chi=[]
    for j in range(118):
        chi=0.5*np.sum(np.divide(np.square(test[1:,i]-train[1:,j]),(test[1:,i]+train[1:,j])))
        distance_chi.append(math.sqrt(chi))
    
    distance_chi=np.array(distance_chi)
    predicted.append(train[0,np.argmin(distance_chi)])

classification_error= np.count_nonzero(predicted==test[0,:])

results["chi"]=(60-classification_error)/60

predicted=[]
for i in range(60):
    distance_chebyshev=[]
    for j in range(118):
        chebyshev=sc_dist.chebyshev(test[1:,i],train[1:,j])
        distance_chebyshev.append(chebyshev)
    
    distance_chebyshev=np.array(distance_chebyshev)
    predicted.append(train[0,np.argmin(distance_chebyshev)])

classification_error= np.count_nonzero(predicted==test[0,:])

results["chebyshev"]=(60-classification_error)/60


print(results)





# for j in range(364):
#     min_error=[]
#     print(j)
#     for i in range(156):
#         m=(Atest[:,i].T).dot(check_norm[:,:j])
#         reconstructed= np.array([average_image.T  +check_norm[:,:j].dot(m.T)]*364).T
#         min_error.append(np.argmin(np.linalg.norm(np.absolute(training-reconstructed),axis=0)))
        
        

#     error_min_ref= [math.ceil(x/7) for x in min_error]
#     reference=[math.ceil((x)/3) for x in range(1,157)]

#     good_prediction=0
#     for i in range(156):
#         if (error_min_ref[i]==reference[i]):
#             good_prediction+=1

#     prediction.append(good_prediction)