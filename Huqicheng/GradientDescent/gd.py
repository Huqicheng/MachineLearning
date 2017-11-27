import sys
import read_data as rd
import numpy as np
import numpy.linalg
import math,sys
import os
import matplotlib.pyplot as plt

#get data from file
#file paths
data_x_path = 'data_x.txt'
data_y_path = 'data_y.txt'
#read data from file
x = rd.read_x(data_x_path)
y = rd.read_y(data_y_path)

#convert to matrix or array
mat_y = np.asmatrix(y).T
print "results of training samples"
print mat_y
mat_x = np.asmatrix(x)
print "training sample:"
print mat_x
arr_x = np.asarray(mat_x)

#get dimension
dim = mat_x.shape[1]

#initiate theta
theta = np.matrix(np.zeros((1,dim)))
thetaT = theta.T

#initiate learning rate
alpha = 0.05

#initiate ite_time
ite_time = 20

while (True):
    mat_result = mat_x*thetaT
    mat_loss = mat_result-mat_y
    arr_loss = np.asarray(mat_loss)
    
    #tile the array
    arr_gradient = np.tile(arr_loss,(1,dim)) #repeat (1,dim)
    arr_gradient = np.multiply(arr_gradient,arr_x)
    gradient = np.sum(arr_gradient,axis=0)
    gradient = np.asmatrix(gradient)
    
    #descending
    thetaT2 = thetaT - alpha*gradient.T

    #convergence condition
    if(numpy.linalg.norm(thetaT2.T-thetaT.T)<=0.0001):
        break;
    
    thetaT = thetaT2
    

print ("theta:%s")%(str(thetaT.T))
print mat_x*thetaT

x = [l_x[1] for l_x in np.asarray(mat_x)]

x = np.asarray(x)

y = np.asarray((mat_x*thetaT).T)[0]


y_dot = [l_y[0] for l_y in np.asarray(mat_y)]
p=plt.subplot(111)
p.axis([0.0,5.01,-1.0,5.5])
p.plot(x,y,"g-",label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
p.plot(x,y_dot,"r*",label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
plt.show()
