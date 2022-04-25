# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:41:25 2021

@author: AnaMaria
"""

import pandas as pd
import numpy as np

a=pd.read_excel("BaseDatos.xlsx",dtype=int) 

def minmax_norm(df):
   
    return (df - df.min()) / ( df.max() - df.min())

df_minmax_norm = minmax_norm(a)
df_minmax_norm = df_minmax_norm.fillna(0)
df_minmax_norm.to_excel("BaseDatos_Normalizada.xlsx")