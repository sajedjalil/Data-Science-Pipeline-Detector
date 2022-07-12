# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import keras as K
from keras.constraints import max_norm
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

def load_data():
    print("Using Python version " + str(py_ver))
    print("Using Keras version " + str(k_ver))
    print("Using TensorFlow version " + str(tf_ver))

    train_data = pd.read_csv('../input/train.csv')

    target_var = 'Cover_Type'

    #数据集的特征
    features = list(train_data.columns)
    features.remove('Id')
    features.remove('Cover_Type')
    #print(features)
    #目标变量的类别
    Class = train_data[target_var].unique()
    #目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))

    #增加一列target,将目标变量进行编码
    train_data['target'] = train_data[target_var].apply(lambda x: Class_dict[x])
    #print(train_data['target'])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(train_data['target'])
    y_bin_labels = []  # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        train_data['y' + str(i)] = transformed_labels[:, i]
    train_features = train_data[features]
    train_label = train_data[y_bin_labels]
    # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(train_data[features], train_data[y_bin_labels], \
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_features,train_label,train_x, test_x, train_y, test_y, Class_dict


def main():

    # 0. 开始
    print("\nforest dataset using Keras/TensorFlow ")
    np.random.seed(4)
    tf.set_random_seed(13)

    # 1. 读取CSV数据集
    print("Loading train_data into memory")
    train_features, train_label,train_x, test_x, train_y, test_y, Class_dict = load_data()

    # 2. 定义模型
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    maxnorm = 5.0
    #sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=128, input_dim=54, kernel_initializer='random_uniform', activation='relu',kernel_constraint=max_norm(maxnorm),
                             bias_constraint=max_norm(maxnorm)))
    model.add(K.layers.Dense(units=256, kernel_initializer='random_uniform',bias_initializer='zeros', activation='relu',
                             kernel_constraint=max_norm(maxnorm),bias_constraint=max_norm(maxnorm)))
    for i in range(6):
        model.add(K.layers.Dense(units=512, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu',
                                kernel_constraint=max_norm(maxnorm),bias_constraint=max_norm(maxnorm)))
    model.add(K.layers.Dense(units=256, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu',
                             kernel_constraint=max_norm(maxnorm),bias_constraint=max_norm(maxnorm)))
    model.add(K.layers.Dense(units=128, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu',
                             kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm)))
    model.add(K.layers.Dense(units=7, kernel_initializer='random_uniform',bias_initializer='zeros', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['categorical_accuracy'])


    # 3. 训练模型
    b_size = 1024
    max_epochs = 250
    print("Starting training ")
    h = model.fit(train_features, train_label, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
    print("Training finished \n")

    # 4. 评估模型
    #eval = model.evaluate(test_x, test_y, verbose=0)
    #print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
    #      % (eval[0], eval[1] * 100) )


    # 5. 使用模型进行预测
    test_data = pd.read_csv('../input/test.csv')
    testid = test_data['Id']
    test_data = test_data.values
    species_dict = {v: k for k, v in Class_dict.items()}

    testlabels = []
    for item in range(len(test_data)):
        #print(item)
        predicted = model.predict(test_data[item,1:].reshape(1,54))
        testlabels.append(species_dict[np.argmax(predicted)])
    df = pd.DataFrame({"Id": testid,"Cover_Type": testlabels})
    columns = ['Id','Cover_Type']
    df.to_csv("output.csv", index=False,columns=columns)

main()

