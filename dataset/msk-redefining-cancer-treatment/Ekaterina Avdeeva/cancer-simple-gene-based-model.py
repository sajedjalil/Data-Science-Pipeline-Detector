import pandas as pd
import numpy as np

# read training and test variants files
# text is not used in this model
df=pd.read_csv('../input/training_variants')
dfTest=pd.read_csv('../input/test_variants')

# Function that computes probabilities
# for a DataFrame to belong to each class
def ComputeProbabilities(df):
    Pclass=[0,0,0,0,0,0,0,0,0,0]
    Nclass=[0,0,0,0,0,0,0,0,0,0]
    vc = df['Class'].value_counts()
    N = vc.size
    countTotal=0
    for i in range(N):
        cl = vc.index[i]
        count = vc.values[i]
        countTotal=countTotal+count
        Nclass[cl]=count
    for icl in range(10):
        Pclass[icl]=Nclass[icl]/countTotal
    return Pclass

# base probabilities computed 
# based on the whole training DataFrame
Pbase=ComputeProbabilities(df)

# the result is (rounding to the second digit):
# Pbase = [0.00,0.17,0.14,0.03,0.21,0.07,0.08,0.29,0.01,0.01]
# 0th element is dummy

# number of entries in the test set
Ntest = dfTest['ID'].count()

# introduce a submission DataFrame
strSubmColumns="ID,class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
dfSubm = pd.DataFrame(columns=strSubmColumns)

# if for a given gene in the test set no genes in the training set found,
# the base probabilities are returned
Parr=Pbase
for i in range(Ntest):
    ID=int(dfTest.loc[i]['ID'])
    strGene = dfTest.loc[i]['Gene']
    if (df[df['Gene']==strGene]['ID'].count()==0): pass
    else:
        # Applying ComputeProbabilities of a 
        # subDataFrame with a given 'Gene' name
        Parr=ComputeProbabilities(df[df['Gene']==strGene])
    dfSubm.loc[i] = pd.Series({'ID':ID,'class1':Parr[1],'class2':Parr[2],'class3':Parr[3],
                               'class4':Parr[4],'class5':Parr[5],'class6':Parr[6],
                               'class7':Parr[7],'class8':Parr[8],'class9':Parr[9]})	

# Prepare submission file
dfSubm['ID']=dfSubm['ID'].apply(lambda n: int(n))
dfSubm.to_csv('submitGeneBased.csv', index=False)
