# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:01:05 2018

@author: Automatize

Cálculo de entropia e Information Gain
"""

import pandas as pd
import numpy as np

def calc_entropy(df, feature, feature_class, value = None):

    df_tmp = df
    
    if value != None:
        df_tmp = df[df[feature] == value]
    
    cnt_total = len(df_tmp)
    
    cnt = len(df_tmp)/len(df)
    
    cnt_true = len(df_tmp[df_tmp[feature_class] == 'Yes'])
    cnt_false = len(df_tmp[df_tmp[feature_class] == 'No'])

    p_true = cnt_true / cnt_total
    p_false = cnt_false / cnt_total

    entropy = -p_true*np.log2(p_true) - (p_false*np.log2(p_false))

    return entropy, cnt, p_true, p_false    

df = pd.read_csv('play_tenis.csv')

#entropia = -p0.log2(p0) - p1.log2(p1)


entropy, cnt,P0, P1 = calc_entropy(df, 'play', 'play')

entropy_w, cnt_w, pw0, pw1 = calc_entropy(df, 'wind', 'play', 'Weak')
entropy_s, cnt_s, ps0, ps1 = calc_entropy(df, 'wind', 'play', 'Strong')

Gain_Wind = entropy - (cnt_w * entropy_w) - (cnt_s * entropy_s)