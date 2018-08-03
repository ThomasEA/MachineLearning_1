# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:56:37 2018

@author: evert
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def hypothesis(X, thetas):
    return np.dot(X, thetas.T)

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def cost_function(X, y, h, theta):
    N = len(X)
    
    #Soma vetorizada
    soma = np.sum((h(X, theta) - y) ** 2.)
    
    return (1./(2. * float(N))) * soma

def gradient_descent(X, y, thetas, alpha, max_iter):
    costs = np.zeros(max_iter, dtype=np.float64)
    N = len(X)
    J = len(thetas)
    soma = 0.
    thetas_tmp = thetas
    
    prev_cost = np.inf
    
    for i in range(max_iter):
        for j in range(J):
            for n in range(N):
                soma += (hypothesis(X[n], thetas) - y[n]) * X[n][j]
    
            thetas_tmp[j] = thetas_tmp[j] - (alpha/len(X)) * soma
        
        thetas = thetas_tmp
        #thetas = thetas - (alpha/len(X)) * np.sum((hypothesis(X, thetas) - y) * X)
        cost = cost_function(X, y, hypothesis, thetas)
        
        if cost < prev_cost:
            costs[i] = cost
            prev_cost = cost
        else:
            break
        

    return thetas, costs

X = np.array([
        [1,21,3],
     [1,16,3],
     [1,24,3],
     [1,14,2],
     [1,30,4]], dtype=np.float32)

y = np.array([399,
     329,
     369,
     232,
     539], dtype=np.float32)



max_iter = 10
alpha = 0.0001
thetas = np.ones(X.shape[1])

cost = cost_function(X, y, hypothesis, thetas)

theta_final, costs = gradient_descent(X, y, thetas, alpha, max_iter)

plt.plot(np.arange(max_iter), costs, c='r')

to_predict = np.array([
        [1,19,4],
     [1,15,3],
     [1,14,3],
     [1,13,3]], dtype=np.float32)

y_real = np.array([299,
     314,
     198,
     212], dtype=np.float32)

predicted = np.zeros(len(y_real))
    
for p in range(len(to_predict)):
    predicted[p] = hypothesis(to_predict[p], theta_final)