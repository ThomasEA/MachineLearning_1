# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import numpy as np
import pandas as pd

def is_impar(x):
    if x % 2 == 0:
        print x, ' é par'
    else:
        print x, ' é impar'
    
def maiorNumero():
    lst = [ -10, 10, 0, 1, 1, 7, 11, 5, 4, 3 ]
    maior = lst[0]
    for x in lst:
        if x > maior:
            maior = x
    print 'Maior número da lista: ', maior

def somaNumerosLista():
    lst = [ -10, 10, 0, 1, 1, 7, 11, 5, 4, 3 ]
    soma = 0
    for x in lst:
        soma += x
    print 'Soma números da lista: ', soma

def fibonacci(max):
    vet = [0,1]
    for x in range(max):
        vet.append(vet[-1] + vet[-2])
    print 'Sequência Fibonacci: ', vet

def _99_bottles_of_beer():
    sequencia = ''
    for x in range(99, 0, -1):
        if x == 1:
            print x,' bootle of beer on the wall', x, 'bottle of beer.'
            sequencia = 'no more beer on the wall.'
        else:
            print x,' bootles of beer on the wall', x, 'bottles of beer.'
            sequencia = str(x - 1) +' bootles of beer on the wall'
        print "Take one down, pass it around,", sequencia

def np_maior_numero_array():
    lst = [ -10, 10, 0, 1, 1, 7, 11, 5, 4, 3 ]
    x = np.max(lst)
    print 'Maior número da lista usando NumPy: ', x

def np_soma_array():
    lst = [ -10, 10, 0, 1, 1, 7, 11, 5, 4, 3 ]
    x = np.sum(lst)
    print 'Soma números da lista usando NumPy: ', x
    
def calc_sd(arr):
    soma = np.sum(arr)
    cnt = np.size(arr)
    media = soma / float(cnt)

    somatorio = 0;
    
    for x in arr:
        somatorio += (x - media) ** 2
    
    print 'Variância: ', 1/float(cnt)*somatorio
    
    resultado = np.sqrt(1/float(cnt)*somatorio)
    
    print 'Desvio padrão: ', resultado

_99_bottles_of_beer()
"""
is_impar(10)
maiorNumero()
somaNumerosLista()
fibonacci(15)

np_maior_numero_array()
np_soma_array()
calc_sd([ -10, 10, 0, 1, 1, 7, 11, 5, 4, 3 ])

"""
