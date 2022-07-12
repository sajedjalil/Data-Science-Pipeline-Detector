# %% [code]
import pandas as pd
import numpy as np
# Data is the name of the data frame and column_names has got to be the list with names of columns
def fit(data,column_name):
    global column_names
    global main_dic
    column_names=column_name
    main_dic={}
    for i in column_names :
        dic={}
        count=0
        for j in data[i].unique():
            dic[j]=count
            count+=1
        main_dic[i]=dic
# This is to do fit_transform on the current data
def fit_transform(data,column_names):
    dic=fit(data,column_names)
    for i in column_names :
        data[i].replace(dic[i],inplace=True)
# Function for replacements
def replacement(data,name,dictionary):
    x=data[name].values
    p=[]
    for i in x :
        if i in dictionary :
            p.append(dictionary[i])
        else:
            p.append(-1)
    data[name]=p
        
# This is to transform new data
def transform(test):
    dic=main_dic
    for i in column_names:
        replacement(test,i,dic[i])
    
    
    

# %% [code]
