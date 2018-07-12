# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:39:46 2018

@author: alu2015111446
"""

import numpy as np
import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.ylabel('Some numbers')
plt.show()


plt.plot(np.array([1,2,3,4]))
plt.ylabel(u'números')
plt.plot([1,2,3,4], [1,4,9,16])
plt.show()

plt.plot(np.array([1,2,3,4]), np.array([1,4,9,16]), 'ro') # c= 'r', marker = 'o'
plt.axis([0,6,0,20]) #[xmin, xmax, ymin, ymax]


t = np.arange(0.,5.,0.2)
plt.plot(t, t, 'r--',
         t, t**2, 'bo',
         t, t**3, 'g^') # c= 'r', marker = 'o'
plt.axis([0,6,0,20]) #[xmin, xmax, ymin, ymax]