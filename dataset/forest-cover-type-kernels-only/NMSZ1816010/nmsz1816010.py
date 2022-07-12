#!/usr/bin/python2.6  
# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
## 
#PATH=r"E:\machine learning project\dataset"
PATH='../input/'
train=pd.read_csv('{}/train.csv'.format(PATH),nrows=None) 
test=pd.read_csv('{}/test.csv'.format(PATH),nrows=None)
y=train['Cover_Type'] #单独将训练的真实标签提取出来
train.drop('Cover_Type',axis=1,inplace=True) #将训练目标从训练集中删除
train.drop('Id',axis=1,inplace=True) #删除文档中的Id列，axis=1表示按列来删除，inplace=True表示在原数据上直接删除
test.drop('Id',axis=1,inplace=True)
##将数据集中的独热编码(one-hot encode)转换为标签编码(label encode)来方便使用基于树的分类器进行分类
##引入一个函数来进行这个操作(函数引用与他处)
def convert_OHE2LE(df):
    tmp_df=df.copy(deep=True) #将df复制给一个新变量，deep=True表示创建了一个新变量，与原变量空间不同
    for s_ in ['Soil_Type','Wilderness_Area']:
        cols_s_=[f_ for f_ in df if f_.startswith(s_)] #目的是找到数据集中开头的标签为s_的那些列，startswith方法是用来检测那些字符串从开头开始符合给定的模式
        sum_ohe=tmp_df[cols_s_].sum(axis=1).unique() #对每个由one-hot编码的数据列进行按列求和，然后unique方法表示，当list有重复值时，只返回一个
            ##如果sum_ohe中存在0,说明one-hot编码是不完整的，需要改进
        if 0 in sum_ohe:
            print('The OHE in {} is incompleted.A new column will be added before label encodeing'.format(s_))
            col_dummy=s_+'_dummy' #新加入的属性列命名为原有的名称后加
            tmp_df[col_dummy]=(tmp_df[cols_s_].sum(axis=1)==0).astype(np.int8)
            cols_s_.append(col_dummy)
            #print(cols_s_)
            sum_ohe=tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                print("The category completion did not work")
        tmp_cat=tmp_df[cols_s_].idxmax(axis=1) #返回frame每一行(对应轴axis=1)的最大值对应的标签(label)
        print(tmp_cat)
        tmp_df[s_+'_LE']=tmp_cat.str.replace(s_,'').astype(np.uint16) # idxmax方法是返回第一个最大值出现的标签，axis=1表示按列进行；
        tmp_df.drop(cols_s_,axis=1,inplace=True)
    return tmp_df

def train_test_apply_func(train_,test_,func_):
    xx=pd.concat([train_,test_]) #将训练集和数据集连接起来
    xx_func=func_(xx)
    train_=xx_func.iloc[:train_.shape[0],:] #iloc的作用是根据标签的位置，来从0开始选取对应列的数据。这里选的是训练集对应的列
    test_=xx_func.iloc[train_.shape[0]:,:]
    del xx,xx_func #删除变量
    return train_,test_
##编码转换函数convert_OHE2LE()
train_x,test_x=train_test_apply_func(train,test,convert_OHE2LE) #将数据集进行编码转化


###特征工程,对特征进行预处理，并且尽量让训练集与测试集同分布
#分析各个特征之间的相关性，由于决策树只能处理单变量(在分支上只考虑单个属性),分类边界是轴平行的，所以尽量使得属性相关度高一些
###对特征进行预处理的函数
def preprocess(df_):
    ##调整Elevation特征的数值,使得决策树可以沿着坐标轴平行的直线来进行分类
    #df_['fe_E_Min_02HDtH']=df_['Elevation']-df_['Horizontal_Distance_To_Hydrology']*0.2  
    ##计算出距离水源的距离(将水平和竖直距离直接转换为直接距离）
    df_['fe_Distance_To_Hydrology']=np.sqrt(df_['Horizontal_Distance_To_Hydrology']**2+df_['Vertical_Distance_To_Hydrology']**2) 
    ##对山影灰度进行处理
    df_['fe_Hillshade_Mean'] = (df_['Hillshade_9am'] + df_['Hillshade_Noon'] + df_['Hillshade_3pm'])/3
    ###将40种土壤类型(Soil_Type)转换为对应的气候特征和地质特征，从而将其分解为两个特征取值数目较少的两个特征
    climatic_zone={}
    geologic_zone={}
    for i in range(1,41):
        if i<=6:
            climatic_zone[i]=2
            geologic_zone[i]=7
        elif i<=8:
            climatic_zone[i]=3
            geologic_zone[i]=5
        elif i == 9:
            climatic_zone[i]=4
            geologic_zone[i]=2
        elif i<=13:
            climatic_zone[i]=4
            geologic_zone[i]=7
        elif i<=15:
            climatic_zone[i]=5
            geologic_zone[i]=1
        elif i<=17:
            climatic_zone[i]=6
            geologic_zone[i]=1
        elif i==18:
            climatic_zone[i]=6
            geologic_zone[i]=7
        elif i<=21:
            climatic_zone[i]=7
            geologic_zone[i]=1
        elif i<23:
            climatic_zone[i]=7
            geologic_zone[i]=2
        elif i<=34:
            climatic_zone[i]=7
            geologic_zone[i]=7
        else:
            climatic_zone[i]=8
            geologic_zone[i]=7
    df_['Climatic_zone_LE']=df_['Soil_Type_LE'].map(climatic_zone).astype(np.uint8)
    df_['Geologic_zone_LE']=df_['Soil_Type_LE'].map(geologic_zone).astype(np.uint8)

    for c in df_.columns:
        if c.startswith('fe_'):
            df_[c]=df_[c].astype(np.float32)#转换为浮点数
    return df_

##利用预处理函数对数据集进行处理
train_x=preprocess(train_x)
test_x=preprocess(test_x)
#print(train_x.head())
###开始利用训练不同的算法进行分类
from sklearn.tree import DecisionTreeClassifier#导入决策树分类器
from sklearn.ensemble import RandomForestClassifier#导入随机森林分类器
from sklearn.metrics import accuracy_score #导入训练器的评估方法，准确率
from sklearn.model_selection import train_test_split#导入模型选择的相关方法，数据分割，
X_train,X_test,y_train,y_test=train_test_split(train_x,y,test_size=0.15,random_state=691,stratify=None)#随机划分训练集和测试集，test_size为测试集样本占原始样本的数目，
print('----------randomforset-----------') #先输出算法的名字
clf=RandomForestClassifier(n_estimators=200,max_depth=1,random_state=314,n_jobs=4,min_samples_split=2,min_samples_leaf=1)
clf_pars={'max_depth':30} 
fit_pars={}
clf.set_params(**clf_pars) #设计估计器的参数，
clf=clf.fit(X_train,y_train,**fit_pars)#根据给定的数据来建立一个学习器，
print('randomforset: train={:.4f},test={:.4f}'.format(accuracy_score(y_train,clf.predict(X_train)),accuracy_score(y_test,clf.predict(X_test))))
####引入一个混淆矩阵的绘制函数(来自sklearn官方文档）
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#计算测试集的混淆矩阵 confusion matrix
clf_cm=confusion_matrix(y_test,clf.predict(X_test))
plt.figure()
plot_confusion_matrix(clf_cm,classes=[1,2,3,4,5,6,7],title='confusion matrix of random forests')
plt.show()
clfs=clf.fit(train_x,y,**fit_pars)

#保存结果
sub=pd.read_csv(PATH+'sample_submission.csv')
sub['Cover_Type']=clfs.predict(test_x)
sub.to_csv('submission_randomforset.csv',index=False)


#####利用决策树进行分类
print('----------decisiontree-----------') #先输出算法的名字
clf=DecisionTreeClassifier(max_depth=1,random_state=315,min_samples_split=2,min_samples_leaf=1)
clf_pars={'max_depth':40} 
fit_pars={}
clf.set_params(**clf_pars) #设计估计器的参数，
clf=clf.fit(X_train,y_train,**fit_pars)#根据给定的数据来建立一个学习器，
print('decisiontree: train={:.4f},test={:.4f}'.format(accuracy_score(y_train,clf.predict(X_train)),accuracy_score(y_test,clf.predict(X_test))))
##计算决策树的混淆矩阵
clf_cm=confusion_matrix(y_test,clf.predict(X_test))
plt.figure()
plot_confusion_matrix(clf_cm,classes=[1,2,3,4,5,6,7],title='confusion matrix of decision trees')
plt.show()
clfs=clf.fit(train_x,y,**fit_pars)
clfs=clf.fit(train_x,y,**fit_pars)

#保存结果
sub=pd.read_csv(PATH+'sample_submission.csv')
sub['Cover_Type']=clfs.predict(test_x)
sub.to_csv('submission_decisiontree.csv',index=False)