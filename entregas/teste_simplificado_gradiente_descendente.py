# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 22:05:48 2018

@author: Everton Thomas e Gustavo Emmel

Regressão multivariada



"""

import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def nan_to_average(X):
    if X.isnull().values.any():
        cols = X.columns.tolist()
        for x in cols:
            col_mean = np.mean(X[x])
            X[np.isnan(X[x])] = col_mean
    
    return X
    
def normalize(X, columns):
    columns[:-1]

    scaled_features = StandardScaler().fit_transform(X.iloc[:,:-1].values)
    scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=columns[:-1])
    scaled_features_df['MEDV'] = X['MEDV']
    X = scaled_features_df
    return X

def add_x0(X):
    x0 = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    X['x0'] = x0
    #reposiciona a última coluna (x0) para a primeira posição
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]
    return X

def scatter_features(x, fx, lblx, lbly, color):
    plt.scatter(x, fx, c=color)
    plt.xlabel('x -> {0}'.format(lblx))
    plt.ylabel('fx -> {0}'.format(lbly))
    plt.show()

def hypothesis(X, theta):
    return np.dot(theta.T, X)

def cost_function(X, fx, h, theta):
    soma = 0.
    N = len(X)
    
    for i in range(N):
        soma += (h(X.iloc[i], theta) - fx.iloc[i]) ** 2.
    
    return (1./(2. * float(N))) * soma

def update_t(X, fx, h, theta, alpha):
    tethas_tmp = [None] * len(thetas)
    
    N = len(X)
    
    soma = 0.
    for j in range(len(theta)):
        for i in range(N):
            #if j == 0:
            #    soma += (h(X.iloc[i], theta) - fx.iloc[i]) * 1
            #else:
            soma += (h(X.iloc[i], theta) - fx.iloc[i]) * X.iloc[i,j]
    
        if (j == 0):
            tethas_tmp[j] = 1
        else:
            tethas_tmp[j] =  theta[j] - ((alpha * (1./float(N))) * soma)
    
    return tethas_tmp

def gradient_descent(X, fx, h, theta, alpha):
    tethas_tmp = [None] * len(thetas)
    
    N = len(X)
    
    soma = 0.
    
    #for i in range(N):
        #if j == 0:
        #    soma += (h(X.iloc[i], theta) - fx.iloc[i]) * 1
        #else:
        #soma += (h(X.iloc[i], theta) - fx.iloc[i]) * X.iloc[i,j]
    
    for j in range(len(theta)):
        tethas_tmp[j] =  theta[j] - ((alpha * (1./float(N))) * np.sum(X.iloc[:,j] * (X @ theta.T - fx), axis=0))
    
    return tethas_tmp


df = pd.read_csv('../datasets/casas/ex1data2.csv', names=['size', 'rooms', 'value'])

#3. adiciona x0 ao dataset para viabilizar o algoritmo
df = add_x0(df)

#4. Terminado o préprocessamento, divide DS em treino e teste
trainset, testset = train_test_split(df, test_size=0.5, shuffle=False)

#5. Avaliamos algumas features e sua relação com a classe
scatter_features(df['size'], df['value'], 'Tamanho', 'Valor casas', 'red')
scatter_features(df['rooms'], df['value'], 'Nº de quartos', 'Valor médio casas', 'red')

X  = trainset.iloc[:,:-1]
fx = trainset.iloc[:,-1]

thetas = np.ones(X.columns.shape[0])
alpha = 0.5

threshold = 0.01
batch_size = 5
epoch = 0.
max_epoch = 10

prev = np.inf
curr = cost_function(X, fx, hypothesis, thetas)

thetas_final = []

while (abs(curr - prev) > threshold) and (epoch < max_epoch):
    bc_cnt = 0 #contador de batch
    
    for i in range(batch_size):
        X_local = X.iloc[bc_cnt:(bc_cnt + batch_size)]
        fx_local = fx.iloc[bc_cnt:(bc_cnt + batch_size)]

        tmp_thetas = gradient_descent(X_local, fx_local, hypothesis, thetas, alpha)
            
        thetas = np.array(tmp_thetas)
        
        bc_cnt += 1
    
    prev = curr
    curr = cost_function(X_local, fx_local, hypothesis, thetas)
    print('custo na época ', epoch, ': ', curr)
    
    if (len(thetas_final) == 0) or (curr < prev):
        thetas_final = thetas
    
    epoch += 1

print('>>> thetas: ', thetas_final)

#Aplicando sobre os dados a serem preditos
X_t = testset.iloc[1, :-1]
fx_t = testset.iloc[1, -1]

#predict = []
val = hypothesis(X_t, thetas)

print('Valor real: ', fx_t, ' / Valor predito: ', val)
#val = cost_function()
#for i in range(len(X_t)):
#    predict.append(hypothesis(X_t.iloc[i], thetas))


