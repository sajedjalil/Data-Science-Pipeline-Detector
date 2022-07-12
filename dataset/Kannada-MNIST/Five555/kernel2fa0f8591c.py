# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib
from scipy import stats

#加载数据并归一化数据
def load_data():
    print('开始加载数据')
    train_data = pd.read_csv(r'/kaggle/input/Kannada-MNIST/train.csv')
    X = train_data.drop('label', axis=1)
    y = train_data['label']

    pca = PCA(n_components=0.7, whiten=True)
    X_train_PCA = pca.fit_transform(X)


    pre_data = pd.read_csv(r'/kaggle/input/Kannada-MNIST/test.csv')
    X_test_final = pre_data.drop('id', axis=1)
    y_test_final = pre_data['id']
    X_pre_PCA = pca.transform(X_test_final)
    print('完成加载数据')
    return X_train_PCA, y, X_pre_PCA,y_test_final

def SaveResult(result,ID,filename):
    output = pd.DataFrame({'id':ID,'label':result})
    output.to_csv('%s.csv' % (filename), index=False)
    #output.to_csv('submission.csv', index=False)

def KNN(X_train_PCA,y_train):
    print('开始训练KNN')
    start = time.time()
    # 把60000个训练样本按照4折的方式划分成训练集和验证集
    folds = 4
    # 把数据拆分成4份
    x_fold = []
    y_fold = []
    x_fold = np.vsplit(X_train_PCA, folds)  # 将数组x_train横向切分成folds份
    y_fold = np.hsplit(y_train, folds)  # 将数组y_train纵向切分成folds份
    # 对当前的k值进行交叉验证
    acc = []  # 定义一个空的列表，存储4次实验的结果
    KNN = neighbors.KNeighborsClassifier(6, weights='distance')
    for i in range(folds):
        # 4折的方式划分数据集
        X_train = np.vstack(x_fold[:i] + x_fold[i + 1:])
        X_val = x_fold[i]
        Y_train = np.hstack(y_fold[:i] + y_fold[i + 1:])
        Y_val = y_fold[i]
        # 使用KNN算法训练模型

        KNN.fit(X_train, Y_train)
        # 验证改模型的精确度
        Y_validation = KNN.predict(X_val)
        acc_score = accuracy_score(Y_validation, Y_val)
        acc.append(acc_score)

    # 求4次交叉验证的均值
    mean_acc = np.mean(acc)
    print("KNN验证精度均值为%f" % (mean_acc))
    end_train = time.time()
    train_time = end_train - start
    print('KNN训练时间为：%f' % (train_time))
    joblib.dump(KNN, 'KNN_model.pkl')
    '''
        # （1）加载数据完之后得到RF算法.
    KNN = neighbors.KNeighborsClassifier(6, weights='distance')
    # （2）使用KNN算法训练模型.
    KNN.fit(X_train_PCA,y_train)
    ratio_bak = KNN.score(X_train_PCA,y_train)
    print("KNN训练精度:", ratio_bak)
    end_train = time.time()
    train_time = end_train - start
    print('KNN训练时间为：%f' % (train_time))
    joblib.dump(KNN,'knn_model.pkl')
    '''

    '''
    KNN_new = joblib.load('knn.pkl')
    #prediction = KNN.predict(X_pre_PCA)
    prediction = KNN_new.predict(X_pre_PCA)
    #saveResult(prediction, filename='RF_result')
    SaveResult(prediction, ID_pre, filename='KNN_result')
    print("结果保存完毕")  
    '''

def SVM(X_train_PCA,y_train):
    print('开始训练SVM')
    start = time.time()

    # 把60000个训练样本按照4折的方式划分成训练集和验证集
    folds = 4
    # 把数据拆分成4份
    x_fold = []
    y_fold = []
    x_fold = np.vsplit(X_train_PCA, folds)  # 将数组x_train横向切分成folds份
    y_fold = np.hsplit(y_train, folds)  # 将数组y_train纵向切分成folds份
    # 对当前的k值进行交叉验证
    acc = []  # 定义一个空的列表，存储4次实验的结果
    SVM = svm.SVC(C=100.0, kernel='rbf', gamma='auto')
    for i in range(folds):
        # 4折的方式划分数据集
        X_train = np.vstack(x_fold[:i] + x_fold[i + 1:])
        X_val = x_fold[i]
        Y_train = np.hstack(y_fold[:i] + y_fold[i + 1:])
        Y_val = y_fold[i]
        # 使用SVM算法训练模型

        SVM.fit(X_train, Y_train)
        # 验证改模型的精确度
        Y_validation = SVM.predict(X_val)
        acc_score = accuracy_score(Y_validation, Y_val)
        acc.append(acc_score)

    # 求4次交叉验证的均值
    mean_acc = np.mean(acc)
    print("SVM验证精度均值为%f",mean_acc)
    end_train = time.time()
    train_time = end_train - start
    print('SVM训练时间为：%f' % (train_time))
    joblib.dump(SVM, 'SVM_model.pkl')

def XGBoost(X_train_PCA,y_train):
    start = time.time()
    # trainDataSet = np.array(X_train_PCA)
    # trainTargetSet = np.array(y_train)
    # testDataSet = np.array(X_test_PCA)
    # （1）加载数据完之后得到SVM算法.
    print('开始训练XGBoost')
    XGBoost_model = XGBClassifier(learning_rate=0.01,
                          n_estimators=20,  # 树的个数-10棵树建立xgboost
                          max_depth=10,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=1,  # 所有样本建立决策树
                          colsample_btree=1,  # 所有特征建立决策树
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27,  # 随机数
                          slient=0)

    XGBoost_model.fit(X_train_PCA,y_train)
    ratio_bak = XGBoost_model.score(X_train_PCA,y_train)
    print("XGBoost训练精确度:", ratio_bak)
    end_train = time.time()
    train_time = end_train - start
    print('XGBoost训练时间为：%f' % (train_time))
    joblib.dump(XGBoost_model, 'xgboost_model.pkl')

    '''
    prediction = XGBoost_model.predict(X_pre_PCA)
    SaveResult(prediction, ID_pre, filename='XGBoost_result')
    print("结果保存完毕")    
    '''

def RF(X_train_PCA,y_train):
    print('开始训练RF')
    start = time.time()
    #把60000个训练样本按照4折的方式划分成训练集和验证集
    folds = 4
    #把数据拆分成4份
    x_fold = []
    y_fold = []
    x_fold = np.vsplit(X_train_PCA, folds)  # 将数组x_train横向切分成folds份
    y_fold = np.hsplit(y_train, folds)  # 将数组y_train纵向切分成folds份
    #对当前的k值进行交叉验证
    acc = [] #定义一个空的列表，存储4次实验的结果
    RF = RandomForestClassifier(n_estimators=134)
    for i in range(folds):
        #4折的方式划分数据集
        X_train = np.vstack(x_fold[:i] + x_fold[i+1:])
        X_val = x_fold[i]
        Y_train = np.hstack(y_fold[:i] + y_fold[i+1:])
        Y_val = y_fold[i]
        # 使用RF算法训练模型

        RF.fit(X_train,Y_train)
        #验证改模型的精确度
        Y_validation = RF.predict(X_val)
        acc_score = accuracy_score(Y_validation, Y_val)
        acc.append(acc_score)

    #求4次交叉验证的均值
    mean_acc = np.mean(acc)
    print("RF验证精度均值为%f"%(mean_acc))

    end_train = time.time()
    train_time = end_train - start
    print('RF训练时间为：%f' % (train_time))
    joblib.dump(RF, 'RF_model.pkl')

def Training_stage(X_train_PCA, y_train):
    print('开始训练')
    #XGBoost(X_train_PCA, y_train)
    RF(X_train_PCA,y_train)
    SVM(X_train_PCA, y_train)
    KNN(X_train_PCA, y_train)
    print('训练完成')

def union_model(X_pre_PCA):
    KNN_new = joblib.load('/kaggle/input/model1/KNN_model.pkl')
    SVM_new = joblib.load('/kaggle/input/model1/SVM_model.pkl')
    #XGBoost_new = joblib.load('xgboost_model.pkl')
    RF_new = joblib.load('/kaggle/input/model1/RF_model.pkl')

    print('对test.csv数据中的每个样本的类别')
    pre_result = []
    for i in range(5000):
        pre_data = np.array(X_pre_PCA[i]).reshape(1, -1)
        pre_svm = SVM_new.predict(pre_data)
        pre_knn = KNN_new.predict(pre_data)
        pre_rf = RF_new.predict(pre_data)
        #Pre = [pre_svm[i], pre_rf[i], pre_knn[i], pre_xgboost[i]]
        Pre = [pre_svm, pre_rf, pre_knn]
        tmp = stats.mode(Pre)[0][0]
        pre_result.append(tmp)
    result = np.array(pre_result)
    return result

def Predicting_stage(X_pre_PCA, ID_pre):
    print('开始预测')
    result = union_model(X_pre_PCA)
    pre_result = result.flatten()
    SaveResult(pre_result, ID_pre, filename='submission')
    print('预测完成')

def main():
    start_time = time.time()
    X_train_PCA, y_train, X_pre_PCA, ID_pre = load_data()
    #Training_stage(X_train_PCA, y_train)
    Predicting_stage(X_pre_PCA, ID_pre)
    end_time = time.time()
    print('所用时间为：',end_time-start_time)

if __name__ == '__main__':
    main()
