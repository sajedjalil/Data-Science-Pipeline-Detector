# -*- coding: utf-8 -*-
# @Time    : 2017/9/30 8:53
# @Author  : LiYun
# @File    : main_v2.py
'''description:
this method is a simple extention of BingQing Wei's XGboost With Context Label Data (ACC: 99.637%)
it's accuracy is 99.81% when 10% of the training data is used as validtion data
and finally, the whole data is used for training
'''
import os
import gc
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def get_classify_train_data(np_file,csv_file):
    if os.path.exists(np_file) == True:
       temp = np.load(np_file)
       return temp['x_train'],temp['y_train'],temp['label']
    else:
        num_features = 9 #每个 word 取前 5 后 4 个字符来编码
        train=pd.read_csv(csv_file)
        tmp=pd.factorize(train['class'])
        y_train,label=tmp[0].astype(np.int8),tmp[1].values
        num_train=len(y_train)
        train['before']=train['before'].astype(np.str)
        x_train=np.zeros([num_train,num_features],np.int8)
        feature=np.zeros([num_train,7],np.int8)# 人工提取的特征
        list1=('a','e','i','o','u')# 元音
        list2=('+','-','*','//','%')# 数学运算符
        for word,row in zip(train['before'].values,range(num_train)):
            if(len(word)>=num_features):
                for c,col in zip(word[:5],range(5)):
                    x_train[row,col]=ord(c)
                for c,col in zip(word[-4:],range(5,9)):
                    x_train[row,col]=ord(c)
            else:
                for c,col in zip(word,range(num_features)):
                    x_train[row,col]=ord(c)
            feature[row, 3] =len(word) # 统计字符串的长度
            dotflag=0
            for c in word:
                if c.isdigit():feature[row,0]+=1# 统计数字的个数
                if c.isupper():feature[row,1]+=1# 统计大写字母的个数
                if c.isalnum()!=True:feature[row,2]+=1# 统计非字母和数字的个数
                if c in list1:feature[row,4]+=1# 统计元音的个数
                if c=='.': dotflag=1
                elif dotflag==1:#  . 后面跟字母置 1 ，数字置 2，其他置 3
                    dotflag = 0
                    if c.isdigit():feature[row,5]+=10
                    elif c.isalpha():feature[row,5]+=100
                    else:feature[row,5]+=1000
                if c in list2:feature[row,6]+=1# 统计数学运算符的个数

        # 掐头去尾，结合上文 2 单词，下文 1 个单词
        num_train-=3
        y_train=y_train[2:-1]
        x_train=np.concatenate((x_train[:-3],x_train[1:-2],x_train[2:-1],x_train[3:],feature[2:-1]),axis=1)
        np.savez(np_file,x_train=x_train, y_train=y_train, label=label)
        return x_train, y_train, label

def get_classify_test_data(np_file,csv_file):
    test=pd.read_csv(csv_file)
    if os.path.exists(np_file) == True:
       temp = np.load(np_file)
       x_test=temp['x_test']
    else:
        num_features = 9 #每个 word 取前 5 后 4 个字符来编码
        human_feature=7 #人工提取7个特征
        num_test=len(test)
        test['before']=test['before'].astype(np.str)
        x_test=np.zeros([num_test,num_features],np.int8)
        feature=np.zeros([num_test,human_feature],np.int8)# 人工提取的特征
        list1=('a','e','i','o','u')# 元音
        list2=('+','-','*','//','%')# 数学运算符
        for word,row in zip(test['before'].values,range(num_test)):
            if(len(word)>=num_features):
                for c,col in zip(word[:5],range(5)):
                    x_test[row,col]=ord(c)
                for c,col in zip(word[-4:],range(5,9)):
                    x_test[row,col]=ord(c)
            else:
                for c,col in zip(word,range(num_features)):
                    x_test[row,col]=ord(c)
            feature[row, 3] =len(word) # 统计字符串的长度
            dotflag=0
            for c in word:
                if c.isdigit():feature[row,0]+=1# 统计数字的个数
                if c.isupper():feature[row,1]+=1# 统计大写字母的个数
                if c.isalnum()!=True:feature[row,2]+=1# 统计非字母和数字的个数
                if c in list1:feature[row,4]+=1# 统计元音的个数
                if c=='.': dotflag=1
                elif dotflag==1:#  . 后面跟字母置 1 ，数字置 2，其他置 3
                    dotflag = 0
                    if c.isdigit():feature[row,5]+=10
                    elif c.isalpha():feature[row,5]+=100
                    else:feature[row,5]+=1000
                if c in list2:feature[row,6]+=1# 统计数学运算符的个数

        # 开头补上2个单词,结尾补上1个单词，结合上文 2 单词，下文 1 个单词
        x_test = np.concatenate((np.zeros([2,num_features],np.int8),x_test,np.zeros([1,num_features],np.int8)),axis=0)
        feature = np.concatenate((np.zeros([2,human_feature],np.int8),feature,np.zeros([1,human_feature],np.int8)),axis=0)
        x_test=np.concatenate((x_test[:-3],x_test[1:-2],x_test[2:-1],x_test[3:],feature[2:-1]),axis=1)
        np.savez(np_file,x_test=x_test)
    return test, x_test

if __name__=='__main__':
    prehead='data/'
    train_data_csv='en_train.csv'
    classify_train_file='classify_train.npz'
    xgb_model='xgb_model.dat'
    test_data_csv='en_test.csv'
    classify_test_file='classify_test.npz'
    xgb_model2='xgb_model2.dat'
    classify_test_file2='classify_test2.npz'

    # 训练模型
    x_train,y_train,label=get_classify_train_data(prehead+classify_train_file,prehead+train_data_csv)
    print(x_train.shape)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    watchlist = [(dtrain, 'train')]
    param = {
        'eta': 0.3,
        'max_depth':10,
        'objective':'multi:softmax',
        'num_class':len(label),
        'eval_metric':'merror',
        'subsample': 1,
        'colsample_bytree': 1,
        'silent':1,
        'seed':0,
    }
    num_boost_rounds=220
    model = xgb.train(param, dtrain, num_boost_rounds, watchlist,verbose_eval=1,xgb_model=xgb_model)
    print('save model ',xgb_model2)
    pickle.dump(model,open(xgb_model2,'wb'))# 保存模型
    del x_train,y_train
    gc.collect()

    # 预测 test 上的 class
    model = pickle.load(open(xgb_model2, "rb"))
    test,x_test=get_classify_test_data(prehead+classify_test_file,prehead+test_data_csv)
    print(x_test.shape)
    dtest = xgb.DMatrix(x_test)
    pred = model.predict(dtest)
    pred = [label[int(x)] for x in pred]
    test['class']=pred
    test.to_csv(os.path.join(prehead, 'test_pred_class.csv'))






























