# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb

print ("Hi this is for testing")

y =np.random.rand(10,10) 

df = pd.DataFrame(y)

df.to_csv("sample.csv")
