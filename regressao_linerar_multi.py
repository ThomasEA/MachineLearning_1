# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:32:12 2018

@author: ALU2015111446
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/casas/ex1data2.csv', names=['size', 'rooms', 'value'])

trainset, testset = train_test_split(df, test_size=0.2)

clf = linear_model.LinearRegression(normalize=True)

xs = trainset[['size', 'rooms']]
fx = trainset[['value']]

test_xs = trainset[['size', 'rooms']]
test_fx = trainset[['value']]

clf.fit(xs, fx)
predictions = clf.predict(test_xs)
score = clf.score(test_xs, test_fx)
print('Intercept (theta 0)...: ', clf.intercept_)
print('Thetas....: ', clf.coef_)
print('Score..: ', score)

x1 = plt.scatter(test_xs[['size']], test_fx, c='red')
x2 = plt.scatter(test_xs[['size']], predictions, c='blue')
plt.legend((x1,x2),('Real', 'Predito'))
plt.xlabel('x')
plt.ylabel('f(x)')

x1 = plt.scatter(test_xs[['rooms']], test_fx, c='red')
x2 = plt.scatter(test_xs[['rooms']], predictions, c='blue')
plt.legend((x1,x2),('Real', 'Predito'))
plt.xlabel('x')
plt.ylabel('f(x)')





