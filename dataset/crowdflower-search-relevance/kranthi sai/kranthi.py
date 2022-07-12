from pandas import  DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import operator
# finding frequency of word in documents. documents is nd array  of strings(like a list)
def frequency(documents,figname):
    #finding uniue and converting to lower case 
    unique=pd.unique(documents)
    uniquelower=list(s.lower() for s in (list(unique)))
    # Removing punctuations
    #print(len(uniquelower))
    #print (uniquelower)
    tokenizer = RegexpTokenizer(r'\w+')
    punctless=[]
    for s in uniquelower:
        list2=[]
        list2=tokenizer.tokenize(s)
        for lis in list2:
            punctless.append(lis)
    #print(len(punctless))
    
    # Removing stop words
    stop = stopwords.words('english')
    nostop=[]
    for s in punctless:
        if s not in stop:
            nostop.append(s)
    #print (len(nostop))
    #print (nostop)
    # Count of each word 
    dic={}
    for s in nostop:
        if s not in dic:
            dic[s]=1
        else:
            dic[s]=dic[s]+1 
    #print(dic)
    dc_sort = sorted(dic.items(),key = operator.itemgetter(1),reverse = True)
    #print ((dc_sort))
    
    # Print in graph top 10 most frequent item queried
    xvalues=[]
    yvalues=[]
    for i in range(0,10):
        s=dc_sort[i]
        xvalues.append(s[0])
        yvalues.append(s[1])
    #print(xvalues)
    #print(yvalues)
    indexes = np.arange(len(xvalues))
    plt.xlabel(' word')
    plt.ylabel(' frequency')
    plt.title('Figure 1')
    width=0.5
    plt.bar(indexes,yvalues,0.5)
    plt.xticks(indexes + width * 0.5, xvalues)
    plt.savefig(figname)
    plt.show()
    pass

# Use Pandas to read in the training and test data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

print (train.columns)
#print (test.columns)
# Print a sample of the training data
#print(train.head())
'''
print (train.size)
print (test.size)
print (train.shape)
print (test.shape)

print (train.shape[0])
print (test.shape[0])
# Now it's yours to take from here

#print (train['query'].head(10))
#print (train.iloc[2])
#print(pd.unique(train['query']).size)
#df1=(pd.unique(train['query']))
#df2=(pd.unique(test['query']))
# no of unique queries in test
#print(df1.sym_diff(df2))// not working on dataframes may be older version of pandas is installed
#print(np.setdiff1d(df1, df2))
#print(np.setdiff1d(pd.unique(train.columns),pd.unique(test.columns)).size)
# Product title and their unique
print (train.columns)
print (train['product_title'].head(10))

print(pd.unique(train['product_title']).size)

print(np.setdiff1d(pd.unique(train['product_title']),pd.unique(test['product_title'])).size)

print(np.intersect1d(pd.unique(train['product_title']),pd.unique(test['product_title'])).size)
'''
frequency(train['query'],'wordcount')
frequency(train['product_title'].sample(1000),'product title')