# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:27:54 2018

@author: alu2015111446
"""

import numpy as np

print '--- Array ----'

a = np.array([1,2,3,'a','b','c'])
print a.shape
print a

print '--- Matrix -----'

b = np.array([[1,2,3],['a','b','c']])
print b.shape
print b
print b[1,1]

print '--- Teste ---'

b = np.zeros((5,5))
print 'Zeros: ', b

b = np.ones((5,5))
print 'Uns: ', b

b = np.eye((5)) #matriz identidade
print 'Identidade: ', b

b = np.full((5,5), 9.) #o ponto é para tratar como float, senão gera com inteiro
print 'Preenche com o valor 9: ', b

b = np.random.random((4,4)) #matriz randômica 4 x 4 randomizando entre 0 e 1
print b

print '--- Exercícios ---'
a = np.array([[10,20,30,40],[50,60,70,80],[90,100,110,120]])
print a

b = a[:2, 1:3]
print b
b[0, 0] = 77
print a #alterou o valor da matriz a tb

#repare na diferença dos objetos
row1 = a[1, :]
row2 = a[1:2, :]
print row1, row1.shape
print row2, row2.shape

#agora com reshape
print row1.reshape(1,4)

a = np.arange(9).reshape(3,3)
print a
print a*3 #multiplica tudo por 3

#especificando o tipo do array
a = np.array([[10,20],[30,40]], dtype=np.float32)
print a

b = np.array([[50,60],[70,80]], dtype=np.float32)
print b

print a + b
print np.add(a, b)

print a - b
print np.subtract(a, b)

print a * b
print np.multiply(a, b)

print a / b
print np.divide(a, b)

print np.sqrt(a, b)

##########################################

x = np.array([[10,20],[30,40]])
y = np.array([[50,60],[70,80]])
v = np.array([90,100])
w = np.array([110,120])

print x.dot(v)
print np.dot(x,v)

print np.sum(x)
print np.sum(x, axis = 0) #Soma as colunas
print np.sum(x, axis = 1) #Soma as linhas

print x.T #Matriz transposta

k = np.arange(9).reshape(3,3)
print k
print k.T

v = np.array([10,20,30])
print v.T
print np.matrix(v).T

#Concatenacao
y = np.array([[1,2,3]])
z = np.array([[1,2,3]])
print np.concatenate((y, z), axis = 0)

print np.vstack((y,z))
print np.hstack((y,z))