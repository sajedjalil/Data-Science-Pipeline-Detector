# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# -*-coding:utf8 -*-
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
df1=pd.read_csv("../input/train.csv")
dftest=pd.read_csv("../input/test.csv")
data_target=df1['TARGET']
columns=list(df1.columns)[1:-1]
count=[]
for i in range(len(columns)):
    count.append(list(df1[columns[i]]).count(0))
count=np.array(count)
#Find the feature need discredization or not
need_not_=np.array(columns)[np.nonzero(count>=20000)[0]]
need_=np.array(columns)[np.nonzero(count<20000)[0]]
#Discredizate the feature
def Discredizate(DataFrame,need_,need_not_):
    returnData_need_=pd.DataFrame(np.zeros((len(DataFrame),len(need_))),columns=need_)
    for item in need_:
        data=[]
        for da in DataFrame[item]:
            data.append([da])
        y_pred=KMeans(n_clusters=3,random_state=None).fit_predict(np.mat(data))
        returnData_need_[item]=y_pred
    def Discre_not(data):
        for i in range(np.shape(data)[1]):
            data[np.nonzero(data[:,i]!=0)[0],i]=1
            #data[np.nonzero(data[:,i]==0)[0],i]=0
            #print(i)
        return data
    data_not_need_=DataFrame[need_not_].values
    data_not_need_=Discre_not(data_not_need_)
    returnData_need_not_=pd.DataFrame(data_not_need_,columns=need_not_)
    return returnData_need_,returnData_need_not_
df2,df3=Discredizate(df1,need_,need_not_)
df1[need_]=df2;df1[need_not_]=df3
#print(df1)
def trainBYS(dataIn):
    p1=list(dataIn['TARGET']).count(1)
    p0=list(dataIn['TARGET']).count(0)
    def conP(data):
        columns=data.columns
        columns=list(columns)[:-1]
        num_df=pd.DataFrame(np.zeros((6,len(data.iloc[0])-1)),columns=columns,index=['p00','p01','p02','p10','p11','p12'])
        data_num=np.array(data)
        data_num1=data_num[np.nonzero(data_num[:,-1]==1)[0],:]
        data_num0=data_num[np.nonzero(data_num[:,-1]==0)[0],:]
        def count(data10):
            re=[]
            for i in range(np.shape(data10)[1]):
                ret=[]
                for k in range(3):
                    ret.extend([list(data10[:,i]).count(k)])
                re.append(ret)
            return np.array(re).T
        num_df.iloc[0:3]=count(data_num0[:,:-1])
        num_df.iloc[3:]=count(data_num1[:,:-1])
        return num_df
    columns=dataIn.columns
    columns=list(columns)[:-1]
    pNum=pd.DataFrame(np.zeros((6,len(columns))),columns=columns,index=['p00','p01','p02','p10','p11','p12'])
    pNum=conP(dataIn)
    return pNum,p0,p1
pNum,p0,p1=trainBYS(df1)
df=pNum.drop(['ID'],axis=1)
df3=(df.iloc[0:3]+1)/(p0+2)
df4=(df.iloc[3:]+1)/(p1+2)
df.iloc[0:3]=df3;df.iloc[3:]=df4
df_need=df[need_]
df_need_not_=df[need_not_]
df_need_not_.drop(['p00','p02','p10','p12'],axis=0)
print(df_need)
def testing(df_need,df_need_not,need_,need_not_,data,pClass1,df1):#df is conditional probability,data is the test data
    data_need_,data_need_not_=Discredizate(data,need_,need_not_)
    #data_Dis=data[need_]=data_need_.copy();data_Dis[need_not_]=data_need_not_.copy()
    def count012(data):
        returnData=np.zeros((3,len(data)))
        for i in range(3):
            returnData[i,np.nonzero(data==i)[0]]=1
        return returnData
    '''
    def create_pw(frame):
        reFrame=pd.DataFrame(np.zeros((3,len(frame.ix[0]))),columns=frame.columns)
        for item in frame.columns:
            for i in range((3)):
                reFrame[item].iloc[i]=list(frame[item]).count(i)
        return reFrame
    pw=create_pw(df1)/len(df1)
    print(pw)
    '''
    return_target=[]
    for i in range(len(data)):
        need_Vocab=count012(np.array(data_need_.iloc[i]))
        p1=sum(np.array(data_need_not_.iloc[i])*np.array(df_need_not.iloc[1]))+sum(sum(np.array(df_need.iloc[3:])*need_Vocab))+np.log(pClass1)
        #pw_data=np.array(data.iloc[0])
        print(p1)
        '''
        for j in range(3):
            pw_data[np.array(data_Dis.iloc[i])==j]=pw.values[j,np.array(data_Dis.iloc[i])==j]
        p1=p1/np.prod(pw_data)
        '''
        p0=sum(np.array(data_need_not_.iloc[i])*np.array(df_need_not.iloc[0]))+sum(sum(np.array(df_need.iloc[0:3])*need_Vocab))+np.log(1.0-pClass1)
        print(p0)
        return_target.append(p1/p0)
    return return_target
ddd=testing(df_need,df_need_not_,need_,need_not_,dftest,float(p1)/float(len(dftest)),df1)
print(ddd)
ret=pd.DataFrame(np.zeros((len(ddd),2)),columns=['ID','TARGET'])
ret['ID']=range(1,len(ddd)+1)
ret['TARGET']=ddd
ret.to_csv('ret.csv')




















