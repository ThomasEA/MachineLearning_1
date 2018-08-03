# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:42:52 2018

@author: evert
"""
import pandas as pd
import numpy as np
import random

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(theta.T, x)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

#computecost
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def hypothesis(X, theta):
    return np.dot(X, theta.T)

def cost_function(X, fx, h, theta):
    soma = 0.
    N = len(X)
    
    for i in range(N):
        soma += (h(X[i], theta) - fx[i]) ** 2.
    
    return (1./(2. * float(N))) * soma

#gradient descent
def gradientDescent_2(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

my_data = pd.read_csv('datasets/casas/ex1data2.csv', names=['size', 'rooms', 'value'])

#setting the matrixes
X = my_data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,3])

#set hyper parameters
alpha = 0.01
iters = 1000

cost_atu = cost_function(X, y, hypothesis, theta)


#running the gd and cost function
g,cost = gradientDescent_2(X,y,theta,iters,alpha)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)