# -*- coding: utf-8 -*-
# @Author: dejei
# @Date:   2019-01-11 19:02:40
# @Last Modified by:   dejei
# @Last Modified time: 2019-01-11 17:36:42
# 备注：本程序是参考其他训练对象程序的改写程序，测试过程中(自适应优化器)理想情况下的训练轮次为40-70轮，
# 并且本程序尚未优化，有些冗杂的自定义变量，请忽略。
import keras
import numpy as np
import tensorflow as tf
import sklearn as sl
import pandas as pd
import os
import seaborn as sns
import xgboost as  xgb
import matplotlib.pyplot as plt 											#功能导入 并用plt代表
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten 								#Dense全连接层 Dropout损失部分，增加鲁棒性，减少计算量 Flatten压缩2维至1维
from keras.layers import Conv2D,MaxPooling2D								# Conv2D2维卷几层 Maxplloing2D池化层、取最大值 
from keras.callbacks import TensorBoard										#引入TensorBoard
from keras.optimizers import Adam,Nadam,RMSprop,Adagrad,Adadelta
from keras.callbacks import ModelCheckpoint
from keras.callbacks import	EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob
import os
import h5py
import keras
np.random.seed(0)

#step1
#导入数据并展示
def load_data(num_classes):
	os.listdir('../input')
	train_data = pd.read_csv(os.path.join('../input','train.csv'))			#训练集数据传入
	test_data = pd.read_csv('../input/test.csv')							#测试集数据传入

	train_data.head()														#训练集例表信息预览
	test_data.head()														#测试集例表信息预览
#	plt.show()
#	print(train_data.head())
#	print(test_data.head())
	train_data.info()														#训练集数据类型
#	print(train_data.columns)												#训练集列表标签栏
	test_data.info()														#测试集数据类型
#	print(test_data.columns)												#测试集列表标签栏
	train_data.columns
#	print(train_data.shape)													#训练集查看尺寸
	test_data.columns
#	print(test_data.shape)													#测试集查看尺寸
	y_test = test_data.copy()

#step2
#数据预处理																	#创建特征列表
	cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
		'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
		'Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']
	train_data = train_data.reindex(np.random.permutation(train_data.index))#随机排列数据集的索引
	new_train_data = train_data.copy()										#permutation会产生的新的数据集赋给new_train_data 
	print(new_train_data.head())											#随机数据集例表信息预览

	def Normalizetrain_data(train_dataset, cols):							#定义归一化处理函数
	    train_dataset[cols] = (train_dataset[cols] - train_dataset[cols].min())/(train_dataset[cols].max() - train_dataset[cols].min())    
	    return train_dataset
	new_train_data = Normalizetrain_data(new_train_data, cols)				#对随机排列的新的训练数据集进行归一化处理
	new_train_data.head()
	print(new_train_data.head())											#归一化的随机数据集例表信息预览
	y_test = Normalizetrain_data(y_test, cols)								#对测试数据集进行归一化处理
	y_test.head()
	y_test.drop(columns='Id', axis=1, inplace=True)							#删除'Id'列
	y_test.head()

#step3
#处理切分数据集、标签集为训练数据集、验证数据集  训练标签、验证标签
	X = new_train_data.iloc[:,1:-1]											#根据标签的所在位置，从0开始计数，
	X.head()																#选取第1列至倒数第1列作为数据集
	y = train_data['Cover_Type'].copy()										#第56列覆盖类型作为标签集
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0) #自动的切分 训练集 和验证集
#	print('X_train.shape',X_train.shape)
#	print(X_train.shape[0],'train samples')
#	print(X_train.shape[0],'test samples')

#step4
#将标签数据转换成2进制数组
	#y_test = keras.utils.to_categorical(y_test,num_classes)
	lable = preprocessing.LabelBinarizer()
	y = lable.fit_transform(y)												#将数据集标签矩阵二值化
	y_train = lable.fit_transform(y_train)									#将训练集标签矩阵二值化
	y_valid = lable.fit_transform(y_valid)									#将验证集标签矩阵二值化
#	print(y)
#	print(y_valid)
	return X_train, X_valid, y_train, y_valid, y_test, test_data, lable, y, X

#step5
#构建序列式神经网络模型
def build_model(keep_prob,num_classes):
	model = Sequential()
	model.add(Dense(1024, input_dim=54, kernel_initializer='glorot_uniform', activation='relu'))#全连接层输出维度=768输入维度=54 权值初始化方法= 激活函数
	model.add(Dropout(0.5))
	model.add(Dense(512, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation = 'softmax'))					#覆盖类型为1-7的标签输出
	model.summary()
	return model

#step6 
#编译神经网络模型
def train_model(model, learing_rate, nb_epoch, batch_size1, samples_per_epoch, X_train, X_valid, y_train, y_valid ,y_test,test_data,lable,y,X):
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',					#保存  每一轮模型保存
								monitor = 'val_loss',
								verbose = 0,
								save_best_only =True,						#保存最优的那个 
								mode ='min')
	early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0003, patience = 6, verbose = 1, mode = 'min') #过拟合的情况下 提前停止训练 loss不变
	tensorboard = TensorBoard(log_dir = './logs', histogram_freq = 1, batch_size = batch_size1, write_graph = False,
								write_grads = False, write_images = False, embeddings_freq=0, embeddings_layer_names= None, embeddings_metadata=None)
	model.compile(
					loss='categorical_crossentropy',						#损失函数
					#optimizer = keras.optimizers.Adadelta(),				#优化器 自适应的学习
					optimizer=keras.optimizers.Nadam(lr=0.00008),
					metrics = ['accuracy']									#列表，包含评估模型在训练
				 )															#和测试时的网络性能的指标

#step7
#训练神经网络模型
	model.fit(X, y, 
			epochs=nb_epoch,
			batch_size=64, 
			verbose = 2,													#显示日志信息
			validation_data = (X_valid,y_valid),
			callbacks =[checkpoint, early_stop, tensorboard]
			)

#step8
#评估神经网络模型
	score = model.evaluate(X_valid,y_valid,verbose=1)			
	print('Test loss:',score[0])											#得分
	print('Test accuracy:',score[1])										#准确度 
#step9
#使用神经网络模型预测
	prediction = model.predict(y_test) 	
	sub = pd.DataFrame({"Id": test_data.iloc[:,0].values,"Cover_Type": lable.inverse_transform(prediction)})
	sub.to_csv("submission.csv", index=False) 
	sub.head()
	
	return model
	
#step10
#保存神经网络模型
	#model.save('my_model.h5')

#使用训练好的模型
#model = keras.models.load_model('my_model.h5')

def main():
	print('-' * 30)
	print('Parameters')
	print('-' * 30)

	keep_prob = 0.5															#定义损失函数参数
	learing_rate = 0.0001
	nb_epoch = 500															#最大500轮
	samples_per_epoch = 3000
	batch_size1 = 16	
	num_classes = 7
	print('keep_prob = %f', keep_prob)
	print('learing_rate = %f', learing_rate)
	print('nb_epoch = %d',nb_epoch)
	print('samples_per_epoch = %d',samples_per_epoch)
	print('batch_size = %d', batch_size1)
	data = load_data(num_classes )
	model = build_model(keep_prob,num_classes)
	model = train_model(model, learing_rate, nb_epoch, samples_per_epoch, batch_size1, *data)
	print("end!")
if __name__ == '__main__':
	main()
