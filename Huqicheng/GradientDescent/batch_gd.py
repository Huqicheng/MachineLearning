import numpy as np
import matplotlib.pyplot as plt

def write2File():
    with open('filename.txt','a') as file:
        write_str = '%f %f\n'%(float_data1,float_data2)
        file.write(write_str)

def readFile(path):
    a = np.loadtxt(path)
    return a

def splitData(data):
    nDim = data.shape[1]-1
    data_x = data[:,0:nDim]
    data_y = data[:,nDim]
    return data_x,data_y

def calcGradient(w,dataX,dataY):
    wT = np.transpose(w)
    # w.T*X - Y
    loss = np.dot(wT,dataX)-dataY
    # loss*X.T
    grad = np.dot(loss,np.transpose(dataX))
    return grad

'''
    batch gradient descent
'''
def BatchGD(dataX,dataY,learning_rate=0.05):
    nDim = dataX.shape[0]
    w = np.zeros((nDim,1))
    preW = w
    while (True):
        # calc gradient
        grad = calcGradient(w,dataX,dataY)
        
        # w = w - lr*grad
        w = w - np.transpose(learning_rate*grad)
        
        # convergence
        if(np.linalg.norm(w-preW)<=np.finfo(float).eps):
            break
        
        preW = w
    return w


if __name__=="__main__":
    #  output of readFile is an ndarray
    data = readFile('data.txt')
    
    # split data
    data_x,data_y = splitData(data)
    data_x = np.transpose(data_x)
    
    w = BatchGD(data_x,data_y)
    print w

    wT = np.transpose(w)
    
    # plot
    y_dot = [l_y for l_y in data_y]
    x = [l_x[1] for l_x in np.asarray(np.transpose(data_x))]
    y = [l_y for l_y in np.asarray(np.dot(wT,data_x))[0]]
    
    p=plt.subplot(111)
    p.axis([0.0,5.01,-1.0,5.5])
    p.plot(x,y,"g-",label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
    p.plot(x,y_dot,"r*",label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
    plt.show()
    
    print y_dot



