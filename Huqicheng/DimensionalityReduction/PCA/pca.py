import numpy as np


# centralization
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal
    return newData,meanVal

def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)
    sortArray=sortArray[-1::-1]
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca(dataMat,percentage=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    n=percentage2n(eigVals,percentage)
    # get indices
    eigValIndice=np.argsort(eigVals)

    # get values in descending order
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    
    # get the vector
    n_eigVect=eigVects[:,n_eigValIndice]
    
    # to lower dimension
    lowDDataMat=newData*n_eigVect
    
    # reconstruct data
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal
    return lowDDataMat,reconMat
