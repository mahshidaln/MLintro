import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def fit_LinRegr(X_train, y_train):
    #Add implementation here
    N , d = X_train.shape   #dimension of X_train
    w = np.zeros((d+1,))    #initializing w
    offset = np.ones((N,1)) #vector of 1 as the offset
    X = np.hstack((offset, X_train))     #adding offset to X_train
    XT = np.transpose(X)    #calculating w using least squares solution
    XTX = np.dot(XT, X)
    XTY = np.dot(XT, y_train)
    w = np.dot(np.linalg.inv(XTX), XTY)
    return w

def mse(X_train,y_train,w):
    #Add implementation here
    N , d = X_train.shape
    offset = np.ones((N,1))
    X = np.hstack((offset, X_train))
    sq = 0
    for i in range(N):  #iterating over the samples to compute squared error
        WTX = np.dot(np.transpose(w), X[i])
        sq += (WTX - y_train[i])**2
    avgError = sq/N #computing the mean of squared error
    return avgError

def pred(X_train,w):
    #Add implementation here
    Y_pred = np.dot(np.transpose(w), X_train) #y_pred = WT . X
    return Y_pred

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here
    lr = linear_model.LinearRegression()
    lr.fit(X_train, Y_train)
    Y_Pred = lr.predict(X_test)
    e = mean_squared_error(Y_test, Y_Pred)
    return e

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    #Testing Part 2a
    w=fit_LinRegr(X_train, y_train)
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

testFn_Part2()

"""
Mean squared error from Part 2a is  2674.3370338831633
Mean squared error from Part 2b is  2674.337033883164

We see that our implementation is working pretty closely to the one from Scikit and we can see a very very small difference between the
the mean squared values we get from the two implementations.
"""