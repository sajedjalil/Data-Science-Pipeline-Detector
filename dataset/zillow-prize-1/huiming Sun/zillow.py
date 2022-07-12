# 没有参考时间变量 因此这个脚本没有考虑时间对logerror的影响
# 没有数据去噪与填充
# 没有考虑特殊情况房屋
# 没有调整参数
# 截断
# 原来已经给出了一部分logerror 标准时间格式提取出来之后，最后直接一一对应的填充

# 提取出标准时间格式

import numpy as np  #线性代数数值计算
import pandas as pd #数据分析模块 numpy的扩展 read_csv函数所在库 Python Data Analysis Library
import lightgbm as lgb #梯度 boosting 框架，使用基于学习算法的决策树 依赖 numpy scipy 
# 基于Histogram的决策树算法
# 带深度限制的Leaf-wise的叶子生长策略
# 直方图做差加速

# 直接支持类别特征(Categorical Feature)

# Cache命中率优化

# 基于直方图的稀疏特征优化

# 多线程优化
import gc #python垃圾回收机制

#注 ：此python程序员有用完一个变量马上进行垃圾回收的习惯

print('Loading data ...')

# 打开两个input csv文件
# 获取train，prop两个文件句柄 分别代表训练集与属性集
# 没有类似异常的处理 有种想加进去的冲动 毕竟不是原生的 。。。 没有 as 那种方法
train = pd.read_csv('../input/train_2016.csv') # pandas库 
prop = pd.read_csv('../input/properties_2016.csv') # pandas库

# zip() 函数是基本函数之一 返回一个元组列表（tuple list） 意思就是返回一个动态的list 但是里面的每一个元素 都是 不可改变的 tuple
# 本质上在 for循环 里的zip函数只是为了生成一群 列号->类型的键值对而已 属于常用方法
# 如果 dtype 为 numpy包里的float64 的话
# 将其转化为 numpy包里的float32 类型
for c, dtype in zip(prop.columns, prop.dtypes):	# zip 函数
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

# 此处参见 API pandas.DataFrame.merge 
# DataFrame.merge(right, how=’inner’, on=None, left_on=None, right_on=None, left_index=False, 
# right_index=False, sort=False, suffixes=(‘_x’, ’_y’), copy=True, indicator=False)
# 将 train表和prop SQL 左联接起来（类似） 
# Parameters
# right : DataFrame
# how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
# left: use only keys from left frame, similar to a SQL left outer join; preserve key order 类似SQL左联接 保持主键顺序
# right: use only keys from right frame, similar to a SQL right outer join; preserve key order
# outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically
# inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys
# on : label or list 此处就是一个label
# Field names to join on. Must be found in both DataFrames. 
# If on is None and not merging on indexes, then it merges on the intersection of the columns by default.
# 返回包括左表中的所有记录和右表中联结字段相等的记录
# 有没有train 中 id 是 prop 中没有的id 造成数据缺失的
# df_train = id-prop -> logerror
train = train.sample(frac=1) 
df_train = train.merge(prop, how='left', on='parcelid')

# 此处参见 API pandas.DataFrame.drop
# DataFrame.drop(labels, axis=0, level=None, inplace=False, errors=’raise’)
# Return new object with labels in requested axis removed. 
# 
# Parameters
# labels : single label or list-like 此处是一个列表
# axis : int or axis name
# level : int or level name, default None
# For MultiIndex
# inplace : bool, default False
# If True, do operation inplace and return None.
# errors : {‘ignore’, ‘raise’}, default ‘raise’
# If ‘ignore’, suppress error and existing labels are dropped.
# New in version 0.16.1.
# Returns:	
# dropped : type of caller

# 大致的讲一下这里的作用 
# 首先 去除了df_train（已经用prop和train SQL左联接过的表） 中 label 为 id.....的数据列 返回对象给 x_train
# 然后 提取 df_train 中 label 为 logerror的 value 返回对象 给 y_train
# x,y这么明显的命名 是不是看出什么了 ？先猜猜 自变量 ，因变量咯
# 去除logerror是可以理解的 因为logerror是因变量 
# 这里不应去掉 transactiondate 的处理 ？ 因为要求 submission里有根据时间 应该降维处理成 月 为基本单位的
# propertyzoningdesc', 'propertycountylandusecode 这两个为何要去掉 ？
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values 

# 此处参见 API pandas.DataFrame.shape
# Return a tuple representing the dimensionality of the DataFrame.
# 返回一个 DataFrame对象的维度元组 并把它print出来
# print(x_train.shape, y_train.shape)

#将 x_train的列 存储为 train_columns对象
train_columns = x_train.columns

# 列表解析
for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True) #只当 x_train[c]为 true时

del df_train; gc.collect() # 删除sample与prop 对象并且通过gc.collect进行垃圾回收 

#定义 切片 为 90000 ？ 为何定义为 90000 ？
split = 90000
# 我决定定义 split 为 80000

# 此处打乱时间顺序再抽取


# 此处开始切片运算符 和 多元赋值 （其实就是两个元组而已 括号它没打）[:split]就是从头开始一直到 split（90000）
# 定义 训练集 train 和 观测集 valid（分别是 0->80000 和 80000->90801(train一共有90801条数据)） ，x，y是 属性 和 logerror 的 关系（自变量，因变量） 
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

# 这里将x_train里面的value全部存储变换为float32类（格式），这个float32是 numpy包里的 ，copy应该是深拷贝，浅拷贝？再查查 
x_train = x_train.values.astype(np.float32, copy=False) 
x_valid = x_valid.values.astype(np.float32, copy=False)

# 参见官方 API ： 
# class lightgbm.Dataset(data, label=None, max_bin=255,
# reference=None, weight=None, group=None, silent=False, feature_name='auto',
# categorical_feature='auto', params=None, free_raw_data=True)
# Parameters:	
# data (string/numpy array/scipy.sparse) – Data source of Dataset. When data type is string, it represents the path of txt file
# label (list or numpy 1-D array, optional) – Label of the data
# max_bin (int, required) – Max number of discrete bin for features
# reference (Other Dataset, optional) – If this dataset validation, need to use training data as reference
# weight (list or numpy 1-D array , optional) – Weight for each instance.
# group (list or numpy 1-D array , optional) – Group/query size for dataset
# silent (boolean, optional) – Whether print messages during construction
# feature_name (list of str, or 'auto') – Feature names If ‘auto’ and data is pandas DataFrame, use data columns name
# categorical_feature (list of str or int, or 'auto') – Categorical features, type int represents index, type str represents feature names (need to specify feature_name as well) If ‘auto’ and data is pandas DataFrame, use pandas categorical columns
# params (dict, optional) – Other parameters
# free_raw_data (Bool) – True if need to free raw data after construct inner dataset

# 训练集和观测集（估计是根据观测集来动态调整训练集关系）
# 以后x_train 应该含有时间数据
d_train = lgb.Dataset(x_train, label=y_train) 


d_valid = lgb.Dataset(x_valid, label=y_valid)

# 此处为调参区
# 因为lightgbm.train（）接受的参数是 以dict为参数的
params = {} # 此处初始化 params 字典 实际上就是一个 K-V 对
params['learning_rate'] = 0.002 # 学习率 为 0.002 
params['boosting_type'] = 'gbdt' # traditional Gradient Boosting Decision Tree dart, Dropouts meet Multiple Additive Regression Trees
params['objective'] = 'regression' # 回归分析
params['metric'] = 'mae'
params['sub_feature'] = 0.5 #次级特征
params['num_leaves'] = 60 # Maximum tree leaves for base learners.
params['min_data'] = 500
params['min_hessian'] = 1

watchlist = [d_valid]

# 参见官方API 
# lightgbm.train(params, train_set, num_boost_round=100, valid_sets=None, 
# valid_names=None, fobj=None, feval=None, init_model=None, feature_name='auto',
# categorical_feature='auto', early_stopping_rounds=None, evals_result=None, verbose_eval=True, learning_rates=None, callbacks=None)
#
# Parameters:参数	
# params (dict) – Parameters for training. 字典类型的训练参数
# train_set (Dataset) – Data to be trained. 要训练的数据集 ‘Dataset’是一个通过 lgb.Dataset()返回的特定类型
# num_boost_round (int) – Number of boosting iterations. 增强迭代次数 类型为 int
# valid_sets (list of Datasets) – List of data to be evaluated during training 训练期间要评估的数据列表 类型为 ...很明显了  
# valid_names (list of string) – Names of valid_sets  valid_sets 的名称 是和 valid_sets一一对应的
# fobj (function) – Customized objective function. 自定义函数
# feval (function) – Customized evaluation function. Note: should return (eval_name, eval_result, is_higher_better) of list of this 自定义评估函数
# init_model (file name of lightgbm model or 'Booster' instance) – model used for continued train 
# feature_name (list of str, or 'auto') – Feature names If ‘auto’ and data is pandas DataFrame, use data columns name
# categorical_feature (list of str or int, or 'auto') – Categorical features, type int represents index, type str represents feature names (need to specify feature_name as well) If ‘auto’ and data is pandas DataFrame, use pandas categorical columns
# early_stopping_rounds (int) – Activates early stopping. Requires at least one validation data and one metric If there’s more than one, will check all of them Returns the model with (best_iter + early_stopping_rounds) If early stopping occurs, the model will add ‘best_iteration’ field
# evals_result (dict or None) –
# This dictionary used to store all evaluation results of all the items in valid_sets. Example: with a valid_sets containing [valid_set, train_set]
# and valid_names containing [‘eval’, ‘train’] and a paramater containing (‘metric’:’logloss’)
# Returns: {‘train’: {‘logloss’: [‘0.48253’, ‘0.35953’, ...]},
#  ‘eval’: {‘logloss’: [‘0.480385’, ‘0.357756’, ...]}}
# passed with None means no using this function
# verbose_eval (bool or int) –
# Requires at least one item in evals. If verbose_eval is True,
# the eval metric on the valid set is printed at each boosting stage.
# If verbose_eval is int,
# the eval metric on the valid set is printed at every verbose_eval boosting stage.
# The last boosting stage
# or the boosting stage found by using early_stopping_rounds is also printed.
# Example: with verbose_eval=4 and at least one item in evals,
# an evaluation metric is printed every 4 (instead of 1) boosting stages.
# learning_rates (list or function) – List of learning rate for each boosting round or a customized function that calculates learning_rate in terms of current number of round (e.g. yields learning rate decay) - list l: learning_rate = l[current_round] - function f: learning_rate = f(current_round)
# callbacks (list of callback functions) – List of callback functions that are applied at each iteration. See Callbacks in Python-API.md for more information.
# Returns:	
# booster
# Return type:	返回一个 训练过的booster模型
# a trained booster model

# 对于给定params（参数）进行训练
# 迭代次数是否可以增加 ？ 会不会过拟合 ？
clf = lgb.train(params, d_train, 450, watchlist) # 训练参数 # 训练的数据集 # 迭代次数 # 训练期间要评估的数据列表

del d_train, d_valid; gc.collect() # 删除d_train与d_valid 对象并且通过gc.collect进行垃圾回收 
del x_train, x_valid; gc.collect() # 删除x_train与x_valid 对象并且通过gc.collect进行垃圾回收 

#预测的准备工作 
print("Prepare for the prediction ...")

#首先读入空的submission样例文件
sample = pd.read_csv('../input/sample_submission.csv')

#更换标签名 程序员小兄弟的强迫症（其实是提交文件的要求 hhhhh）
sample['parcelid'] = sample['ParcelId']

# 返回包括右表中的所有记录和左表中联结字段相等的记录
df_test = sample.merge(prop, on='parcelid', how='left') 
del sample, prop; gc.collect() #删除sample与prop 对象并且通过gc.collect进行垃圾回收 

x_test = df_test[train_columns] 
del df_test; gc.collect() #删除df_test 对象并且通过gc.collect进行垃圾回收 

#用的列表解析
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

# 这里将x_test里面的value全部存储变换为float32类（格式），这个float32是 numpy包里的 ，copy应该是深拷贝，浅拷贝？再查查 
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")
#开始预测
# num_threads > 1 will predict very slow in kernal
#clf 已经经过训练的 boost 模型
#此处参见官方API 
#reset_parameter(params)
#
# Parameters:	
# params (dict) – New parameters for boosters 接受字典作为 参数
# silent (boolean, optional) – Whether print messages during construction 静默状态 ？

clf.reset_parameter({"num_threads":1}) #上面的注释说了 如果开启的线程数大于1的话 在内核上的运行会编的极慢 注：此处内核为 zillow网站上的内核

# 此处参见 官方API
# predict(data, num_iteration=-1, raw_score=False, pred_leaf=False, data_has_header=False, is_reshape=True, pred_parameter=None)
#
# Parameters:	参数
# data (string/numpy array/scipy.sparse) – Data source for prediction When data type is string, it represents the path of txt file
# num_iteration (int) – Used iteration for prediction, < 0 means predict for best iteration(if have) 
# raw_score (bool) – True for predict raw score
# pred_leaf (bool) – True for predict leaf index
# data_has_header (bool) – Used for txt data
# is_reshape (bool) – Reshape to (nrow, ncol) if true
# pred_parameter (dict) – Other parameters for the prediction
# Returns:	
# Return type:	
# Prediction result 返回类型为 预测的结果 和 X_test一样的类型
p_test = clf.predict(x_test) #调用clf.predict()函数 将x_test数据集变成最后的结果集

# 混合模型 请注意这里 为什么这么写 ：The 0.011 value is approximately the mean from the training set
# p_test = p_test * 0.93 + 0.064 * 0.012
del x_test; gc.collect() #删除x_test 对象并且通过gc.collect进行垃圾回收

print("Start write result ...")
#写入文件 首先读入空的submission样例文件
sub = pd.read_csv('../input/sample_submission.csv')
#使用 for in 将sub中的信息（不包括ParceId这列的）
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test #预测结果的存储

        
#转化sub对象为名字为"lgb_starter.csv"的csv文件 保留四位小数格式 
sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')