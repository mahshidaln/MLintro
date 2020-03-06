import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):
    #Add implementation here
    N , d = X_train.shape   #dimension of X_train
    w = np.zeros((d+1,))    #initializing w
    offset = np.ones((N,1)) #vector of 1 as the offset
    X = np.hstack((offset, X_train))    #adding offset to X_train
    for i in range(1000):   #interate T times
      e_prev = errorPer(X,y_train,w)    #calculating the avg error before updating w
      for j in range(N):
        if(pred(X[j], w) != y_train[j]):
            w_new = w + y_train[j]*X[j] #calculating new w if a point is miscalssified
            if(errorPer(X,y_train,w_new) < e_prev): #checking the error with the new value for w and update w if the error has become better
                w = w_new
                break
    return w
    
def errorPer(X_train,y_train,w):
    #Add implementation here
    N , d = X_train.shape
    missclassified = 0
    for i in range(N):
        if(pred(X_train[i], w) != y_train[i]):  #comparing the predicted value with the expected value and find the number of miscalssified points
            missclassified += 1
    avgError = missclassified/N
    return avgError
    
def confMatrix(X_train,y_train,w):
    #Add implementation here
    conf = np.zeros((2,2))
    N , d = X_train.shape
    offset = np.ones((N,1))
    X = np.hstack((offset, X_train))
    for i in range(N):
        y_pred = pred(X[i], w)
        if( y_pred == -1):
            if(y_pred == y_train[i]):
                conf[0][0] += 1 #true negative
            else:
                conf[1][0] += 1 #false negative
        else:
            if(y_pred == y_train[i]):
                conf[1][1] += 1 #true positive
            else:
                conf[0][1] += 1 #false positive
    return conf
    
def pred(X_i,w):
    #Add implementation here
    activation = np.dot(np.transpose(w),X_i) #y_pred = WT . X
    if(activation > 0):
        label = 1
    else:
        label = -1
    return label
    
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here
    pcn = Perceptron()
    pcn.fit(X_train, Y_train)
    Y_pred = pcn.predict(X_test)
    conf_mat = confusion_matrix(Y_test, Y_pred)
    return conf_mat

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
            
    #Testing Part 1a
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)
    
    #Testing Part 1b
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()

"""Confusion Matrix is from Part 1a is:  [[12.  1.]
 [ 0.  7.]]
Confusion Matrix from Part 1b is: [[11  2]
 [ 0  7]]
 We see that the result from implemented Pocket algorithm is pretty close to the one from Scikit. As out of 13 negatives our implementation 
 has recognized 12 true negatives and 1 false positive while the Scikit implementation has recognized 11 true negatives and 2 false positive.
 in this run, both implementations works the same about positive cases.
 """