# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:01:32 2018

@author: Exercício Regressão Linerar
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def hypothesis(x, t0, t1):
    return t0 + t1 * x;

def cost_function(X, fx, h, t0, t1):
    soma = 0.
    N = len(X)
    
    for i in range(N):
        soma += (h(X[i], t0, t1) - fx[i]) ** 2.
    
    return (1./(2. * float(N))) * soma

def update_t0(X, fx, h, t0, t1, alpha):
    """
    Atualiza t0 com base nos N valores passados para esta função.
    """
    
    N = len(X)
    soma = 0.
    
    for i in range(N):
        soma += (h(X[i], t0, t1) - fx[i])
    
    return t0 - ((alpha * (1./float(N))) * soma)


def update_t1(X, fx, h, t0, t1, alpha):
    """
    Atualiza t1 com base nos N valores passados para esta função.
    """
    N = len(X)
    
    soma = 0.
    for i in range(N):
        soma += (h(X[i], t0, t1) - fx[i]) * X[i]
    
    return t1 - ((alpha * (1./float(N))) * soma)

df = pd.read_csv('datasets/regressao_univariada/ex1data1.csv')

df = StandardScaler().fit_transform(df)

trainset, testset = train_test_split(df, test_size=0.5)

X  = trainset[:,0]
fx = trainset[:,1]

#plt.plot(X, [hypothesis(x, t0, t1) for x in X], c='blue')
plt.scatter(X, fx, c='red')
plt.xlabel('x')
plt.ylabel('f(x)')
#plt.title(u'Predições ' + r'para $\theta_0=$' + str(t0) + r' e $\theta_1=$' + str(t1))
plt.show()

t0 = 0.1
t1 = 0.5
alpha = 0.1

threshold = 0.001
batch_size = 2
epoch = 0.
max_epoch = 10

prev = np.inf
curr = cost_function(X, fx, hypothesis, t0, t1)

while (abs(curr - prev) > threshold) and (epoch < max_epoch):
    bc_cnt = 0 #contador de batch
    
    for i in range(batch_size):
        X_local = X[bc_cnt:(bc_cnt + batch_size)]
        fx_local = fx[bc_cnt:(bc_cnt + batch_size)]

        temp0 = update_t0(X_local, fx_local, hypothesis, t0, t1, alpha)
        temp1 = update_t1(X_local, fx_local, hypothesis, t0, t1, alpha)
        
        t0 = temp0
        t1 = temp1
        
        bc_cnt += 1
    
    prev = curr
    curr = cost_function(X, fx, hypothesis, t0, t1)
    print('custo na época ', epoch, ': ', curr)
    epoch += 1

print('t0: ', t0)
print('t1: ', t1)

#Aplicando sobre os dados a serem preditos
X_t = testset[:, 0]
fx_t = testset[:, 1]

predict = []

for i in range(len(X_t)):
    predict.append(hypothesis(X_t[i], t0, t1))

plt.scatter(X_t, fx_t, c='red')
plt.plot(X_t, predict, c='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
#plt.title(u'Predições ' + r'para $\theta_0=$' + str(t0) + r' e $\theta_1=$' + str(t1))
plt.show()
