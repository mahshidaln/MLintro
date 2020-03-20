#!/usr/bin/env python
# coding: utf-8

import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_iris

def activation(s):
    #Enter implementation here
    return np.maximum(0, s) #RELU function

def derivativeActivation(s):
    #Enter implementation here
    if(s >= 0): #derivative of the RELU function
      return 1
    else:
      return 0

def outputf(s):
    #Enter implementation here
    return 1/(1 + np.exp(-s)) #sigmoid function

def derivativeOutput(s):
    #Enter implementation here
    return (outputf(s)*(1-outputf(s))) #derivative of the sigmoid function

def errorf(x_L,y):
    #Enter implementation here
    if (y == 1):  #log loss function
        return -np.log(x_L)
    elif(y == -1):
        return -(np.log(1-x_L))

def derivativeError(x_L,y):
    #Enter implementation here
    if (y == 1):  #derivative of log loss function
        return -1/x_L
    elif(y == -1):
        return 1/(1-x_L)

def errorPerSample(X,y_n):
    #Enter implementation here
    x_L = X[-1][0]  # calculating the error of the network's output
    return errorf(x_L, y_n)

def pred(x_n,weights):
    #Enter implementation here 
    X , S = forwardPropagation(x_n, weights) #getting the output (probability) of the network for test samples
    c = 0
    if (X[-1][0] >= 0.5): # calculating the label based on the probability
        c = 1
    else: 
        c = -1
    return c

def confMatrix(X_train,y_train,w):
    #Enter implementation here
    conf = np.zeros((2,2))
    N , d = X_train.shape
    offset = np.ones((N,1))
    X = np.hstack((offset, X_train)) #adding bias nodes
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

def updateWeights(weights,g,alpha):
    #Enter implementation here
    nW = copy.deepcopy(weights) #make a copy of weights as they have the same dimensionality
    for l in range(len(weights)):
        r = len(weights[l])
        for i in range(r):
            c = len(weights[l][i])
            for j in range(c):
                nW[l][i][j] = weights[l][i][j] - alpha*g[l][i][j] #gradient descent step
    return nW

def backPropagation(X,y_n,s,weights):
    #Enter implementation here
    x_L = X[-1][0] #output
    L = len(X)
    delta = []
    g = copy.deepcopy(weights) #make a copy of weights as they have the same dimensionality
    derror_dxL = derivativeError(x_L, y_n) 
    delta_L = derror_dxL * derivativeOutput(s[-1]) #backward message of the last layer
    delta.insert(0, delta_L)
    for l in reversed(range(1,L-1)): #starting from the last layer
        d_next = len(s[l]) 
        d = len(s[l-1])
        delta_l = []
        for i in range(d):
            delta_l.append(0)
            for j in range(d_next):
               delta_l[i] = delta_l[i] + delta[0][j]*weights[l][i+1][j]*derivativeActivation(s[l-1][i]) #calculating the backward message of hidden layers
        delta.insert(0, delta_l)
    for l in range(L-1):
        d = len(X[l])
        d_next = len(s[l])
        for i in range(d):
            for j in range(d_next):
                g[l][i][j] = delta[l][j]*X[l][i] #derivative of error wrt weights
    return g

def forwardPropagation(x, weights):
    X = [x]
    S = []
    L = len(weights) + 1
    for l in range(L-1):
        sl = np.dot(np.transpose(weights[l]), X[l]) #calculating the s as sum of w*x
        S.append(sl)
        dl1 = sl.shape[0]
        xl = []
        if(l == L-2):
            for i in range(dl1):
                xl.append(outputf(sl[i])) #calculating X in the last layer (the output of the network)
        else:
            xl.append(1)
            for i in range(dl1):
                xl.append(activation(sl[i])) #calculating X in hidden layers
        X.append(xl)    
    return X, S

def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    #Enter implementation here
    N = len(X_train)
    L = len(hidden_layer_sizes) + 2
    layer_sizes = np.insert(hidden_layer_sizes, 0, len(X_train[0])) #adding the size of first and last layer to get the network's architecture
    layer_sizes = np.append(layer_sizes, 1)
    avg_errors = []
    offset = np.ones((N,1)) #vector of 1 as the offset
    X_train = np.hstack((offset, X_train))
    y_train = y_train[:,np.newaxis]
    Xy_train = np.concatenate((X_train, np.array(y_train)), axis=1) #concatenating X and Y in order to shuffle together
    weights = []
    for layer in range(L-1): #initializng the weights with 0.1
        d = layer_sizes[layer]
        d_next = layer_sizes[layer+1]
        weights_l = []
        for p in range(d+1):
            weights_l_p = []
            for q in range(d_next):
                weights_l_p.append(0.1)
            weights_l.append(weights_l_p)
        weights.append(weights_l)
    for e in range(epochs): #go through the training set for #epoch times
        np.random.shuffle(Xy_train)
        X_shuffled = Xy_train[:,:-1] #seperating the X and y after shuffling
        y_shuffled = Xy_train[:,-1]
        error = 0
        for n in range(N): #do training for all the samples in the training set
            X_n = X_shuffled[n] 
            y_n = y_shuffled[n]
            X, S = forwardPropagation(X_n, weights)
            g = backPropagation(X, y_n, S, weights)
            weights = updateWeights(weights, g, alpha) 
            error = error + errorPerSample(X, y_n) #calculating the error for every sample
        avg_errors.append(error/N) #calculating the average error for every epoch    
    return avg_errors, weights

def plotErr(e,epochs):
    #Enter implementation here
    plt.plot(list(range(epochs)), e) #ploting the average error for every epoch
    plt.show()

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Enter implementation here
    mlp = MLPClassifier(hidden_layer_sizes=(300,100), solver='sgd', alpha=0.00001, random_state=1) #instantiating a neural net in scikit
    mlp.fit(X_train, Y_train) #training the net
    Y_pred = mlp.predict(X_test)  #testing the net
    conf_mat = confusion_matrix(Y_test, Y_pred) #calculating the confusion matrix
    return conf_mat

def test():
    
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
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
test()

"""
Confusion Matrix is from Part 1a is:  [[11.  0.]
 [ 1.  8.]]
Confusion Matrix from Part 1b is: [[10  1]
 [ 0  9]]
 As the results show the implmented neural network is working acceptably well as it has made one mistake similar to the one implemented in scikit
 and as the number of epochs increases we can see the decrease in the average error of our network"""