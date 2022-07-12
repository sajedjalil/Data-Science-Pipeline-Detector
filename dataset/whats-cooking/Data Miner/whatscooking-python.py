# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:09:21 2015

@author: Dipayan
"""


from pandas import Series, DataFrame
import pandas as pd


# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%


a=pd.read_json("../input/test.json")
print(a)
