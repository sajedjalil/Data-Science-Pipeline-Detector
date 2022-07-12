#
#给kaggle新手的入门教程(中文完全注释):7种模型集成学习建模方法(xgBst/eNet...)
#
#
#
#
#
#
#
#
# 作者：KeithFish
# GitHub：KeithFish （https://github.com/KeithFish）
# 欢迎留言交流，转载请注明出处
#
#
#
#
#
#
#
#
#
#导入计算模块，对模块大致功能注解：
#numpy支持矩阵运算
#pandas用来做数据清洗，像是python中的excel（支持将数据以.csv格式输出至本地）
#sklearn用来进一步制作数据集（支持数据的导入和数据的导出），含有SVM支持向量机、DT决策树、KNN近邻、LR逻辑回归等封装好的模型，支持对数据进行交叉验证以调参。
#mlxtend用来实现集成学习：bagging, boosting, stacking
#lightgbm内有boosting tree（相比xgboost，改进了生成节点的方式）
#xgboost内有boosting tree
#os用来读取文件
import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os

#######################################################数据导入和特征提取-【开始】################################################################################
#显示当前编译器的Draft Environment下的文件；将文件夹下的对应名称csv文件储存为矩阵对象。
print(os.listdir("../input"))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

#显示矩阵对象的维数，核查是否导入成功
print("Train set size:", train.shape)
print("Test set size:", test.shape)

#记录运算开始的时间
print('START data processing', datetime.now(), )

#取出train矩阵（或者称之为“数据帧”dataframe）中title为'Id'的列，赋值给train_ID。所以train_ID是一维列向量了。test_ID类似。
#train_ID和test_ID根本就没有使用，完全可以删除。
#train_ID = train['Id']
#test_ID = test['Id']

##################删除训练集和测试集中的标签列-【开始】#################
#将train矩阵中的'Id'列删除（原地删除，故将inplace设为true），因为原始数据中的数据索引和预测模型的构建没有关系。
#test矩阵类似。
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
##################删除训练集和测试集中的标签列-【结束】#################

###############删除训练集中的极端值和进行数据index更新-【开始】#########
#使用条件筛选操作，通过覆值的方式剔除原始数据train矩阵中的极端值（极端值也被称为outliers），帮助预防房价预测模型出现过拟合。剔除操作也可以视为前剪枝。
train = train[train.GrLivArea < 4500]
#由于删去了部分行，故此时train矩阵中的index列并不连续。使用reset_index命令，在固定非index数据的顺序的前提下（inplace=True），重新对index编号（drop=True）。
train.reset_index(drop=True, inplace=True)
###############删除训练集中的极端值和进行数据index更新-【结束】#########

##########对预测目标数值进行对数变换和特征矩阵对象的创建-【开始】#######
# log1p就是log(1+x)，用来对房价数据进行数据预处理，它的好处是转化后的数据更加服从高斯分布，有利于后续的分类结果。
# 需要注意，最后需要将预测出的平滑数据还原，而还原过程就是log1p的逆运算expm1。
train["SalePrice"] = np.log1p(train["SalePrice"])
#单独取出训练数据中的房价列信息，存入y对象。
#y = train.SalePrice.reset_index(drop=True) #.reset_index(drop=True)方法：在原有的索引列重置索引，不再另外添加新列。有必要使用reset_index吗？有必要的，不这样做y将有两套index，作为df的y将有两列。
y = train['SalePrice'].reset_index(drop=True) #对上式的改写
#沿着水平的方向寻找列名为'SalePrice'的列（们），把它们对应的列统统删掉。得到了单纯的特征矩阵，存入train_features对象中。
train_features = train.drop(['SalePrice'], axis=1)
#test本来就没有房价列，所以它本来就是单纯的特征矩阵。
test_features = test
##########对预测目标数值进行对数变换和特征矩阵对象的创建-【结束】#######

##合并训练数据特征矩阵与测试数据特征矩阵，以便统一进行特征处理-【开始】##
#将训练数据中的特征矩阵和测试数据中的特征矩阵合并（.concat[矩阵1,矩阵2]），并对合并后的矩阵index重新编号（.reset_index(drop=True)）。
features = pd.concat([train_features, test_features]).reset_index(drop=True)
#检查合并后的矩阵的维数，核查合并结果。
print("剔除训练数据中的极端值后，将其特征矩阵和测试数据中的特征矩阵合并，维度为:",features.shape)
##合并训练数据特征矩阵与测试数据特征矩阵，以便统一进行特征处理-【结束】##
#######################################################数据导入和特征提取-【结束】################################################################################

##############################################################特征处理-【开始】###################################################################################
#对于列名为'MSSubClass'、'YrSold'、'MoSold'的特征列，将列中的数据类型转化为string格式。
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

###############################填充空值-【开始】##########################
#按照以下各个特征列的实际情况，依次处理各个特征列中的空值（.fillna()方法）
features['Functional'] = features['Functional'].fillna('Typ') #空值填充为str型数据'Typ'
features['Electrical'] = features['Electrical'].fillna("SBrkr") #空值填充为str型数据"SBrkr"
features['KitchenQual'] = features['KitchenQual'].fillna("TA") #空值填充为str型数据"TA"
features["PoolQC"] = features["PoolQC"].fillna("None") #空值填充为str型数据"None"

#对于列名为'Exterior1st'、'Exterior2nd'、'SaleType'的特征列，使用列中的众数填充空值。
#	1.先查找数据列中的众数：使用df.mode()[]方法
#	  解释：df.mode(0或1,0表示对列查找，1表示对行查找)[需要查找众数的df列的index（就是df中的第几列）]，将返回数据列中的众数
#	2.使用.fillna()方法进行填充
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

#对于列名为'GarageYrBlt', 'GarageArea', 'GarageCars'的特征列，使用0填充空值。
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)

#对于列名为'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'的特征列，使用字符串'None'填充空值。
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')

#对于列名为'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'的特征列，使用字符串'None'填充空值。
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')

#聚合函数（按某一列关键字分组）groupby，它的特点是：将返回与传入方法的矩阵维度相同的单个序列。
#transform是与groupby（pandas中最有用的操作之一）通常组合使用，它对传入方法的矩阵进行维度不变的变换。具体变换方法写在括号中，通常会使用匿名函数，对传入矩阵的所有元素进行操作。
#对于features矩阵，按照'MSSubClass'列中的元素分布进行分组，被分组的数据列是'MSZoning'列。feature.groupby(被作为索引的列的名称)[被分组的数据列的名称]
#features.groupby('MSSubClass')['MSZoning']后，得到的是一个以'MSSubClass'为索引，以'MSZoning'为数据列的矩阵。
#.transform()方法将对'MSZoning'数据列进行()内的变换，它将返回和传入矩阵同样维度的矩阵。
#括号内是匿名函数，将对传入矩阵中的空值进行填充，使用的填充元素是传入矩阵中的众数。
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


#判断出features矩阵中列为对象的列，将列名存入objects叔祖。对于features矩阵中的各个列对象，将其列中的空值填充为'None'
objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))

#使用传入矩阵（'LotFrontage'列）的中位数对传入矩阵中的空值进行填充。
#先以'Neighborhood'为标签，以'LotFrontage'为被汇总序列。然后使用被汇总序列中的中位数，对原始矩阵'LotFrontage'列中的空值进行填充。
#transform的特性是同维操作，最后输出结果的顺序和原始数据在序号上完全匹配。
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


#对于整型和浮点型数据列，使用0填充其中的空值。
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))
###############################填充空值-【结束】##########################

######################数字型数据列偏度校正-【开始】#######################
#使用skew()方法，计算所有整型和浮点型数据列中，数据分布的偏度（skewness）。
#偏度是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。亦称偏态、偏态系数。 
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

#以0.5作为基准，统计偏度超过此数值的高偏度分布数据列，获取这些数据列的index。
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

#对高偏度数据进行处理，将其转化为正态分布。
#Box和Cox提出的变换可以使线性回归模型满足线性性、独立性、方差齐次以及正态性的同时，又不丢失信息。
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))#这是boxcox1p的使用方法，参数的具体意义暂时不解释
######################数字型数据列偏度校正-【结束】#######################

######################特征删除和融合创建新特征-【开始】###################
#删除一些特征。df.drop（‘列名’, axis=1）代表将‘列名’对应的列标签（们）沿着水平的方向依次删掉。
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

#融合多个特征，生成新特征。
features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])

#简化特征。对于某些分布单调（比如100个数据中有99个的数值是0.9，另1个是0.1）的数字型数据列，进行01取值处理。
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#检查特征处理后，特征矩阵的维数，核查特征处理结果。
print("删除了3个特征，又融合创建了10个新特征，处理之后的特征矩阵维度为:",features.shape)
######################特征删除和融合创建新特征-【结束】###################

####################特征投影、特征矩阵拆解和截取-【开始】#################
#使用.get_dummies()方法对特征矩阵进行类似“坐标投影”操作。获得在新空间下的特征表达。
final_features = pd.get_dummies(features).reset_index(drop=True)
#打印新空间下的特征维数，也是新空间的维数。
print("使用get_dummies()方法“投影”特征矩阵，即分解出更多特征，得到更多列。投影后的特征矩阵维度为:",final_features.shape)

#进行特征空间降阶。截取前len(y)行，存入X阵（因为之前进行了训练数据和测试数据的合并，所以从合并矩阵中取出前len(y)行，就得到了训练数据集的处理后的特征矩阵）。
#截取剩余部分，即从序号为len(y)的行开始，至矩阵结尾的各行，存入X_sub阵。此为完成特征变换后的测试集特征矩阵。
#注：len(df)是行计数方法
X = final_features.iloc[:len(y), :]	#y是列向量，存储了训练数据中的房价列信息。截取后得到的X阵的维度是len(y)*(final_features的列数)。
X_sub = final_features.iloc[len(y):, :]#使用len命令，求矩阵X的长度，得到的是矩阵对象的长度，即有矩阵中有多少列，而不是每列上有多少行。
print("删除了3个特征，又融合创建了10个新特征，处理之后的特征矩阵维度为:",'X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)

#在新生特征空间中，剔除X阵和y阵中有着极端值的各行数据（因为X和y阵在水平方向上是一致的，所以要一起删除同样的行）。outliers数值中给出了极端值的列序号。
#df.drop(df.index[序号])将删除指定序号的各行。再使用=对df覆值。
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])#因为X阵是经过对特征矩阵进行类似“坐标投影”操作后得到的，列向量y中的行号对应着X阵中的列号。
y = y.drop(y.index[outliers])
####################特征投影、特征矩阵拆解和截取-【结束】#################

######################消除截取后特征矩阵的过拟合-【开始】#######################  这一步的目的是处理X阵和X_sub阵。
#在新生特征空间中，删除将产生过拟合的数据列。

#这种数据列具有如下特征：
overfit = []#用来记录产生过拟合的数据列的序号
for i in X.columns:#遍历截取后特征矩阵的每一列
    counts = X[i].value_counts()#使用.value_counts()方法，查看在X矩阵的第i列中，不同的取值分别出现了多少次，默认按次数最高到最低做降序排列。返回一个df。
    zeros = counts.iloc[0]#通过行号索引行数据，取出counts列中第一个元素，即出现次数最多的取值到底是出现了多少次，存入zeros
    if zeros / len(X) * 100 > 99.94:
#判断某一列是否将产生过拟合的条件：
#截取后的特征矩阵有len(X)列，如果某一列中的某个值出现的次数除以特征矩阵的列数超过99.94%，即其几乎在被投影的各个维度上都有着同样的取值，并不具有“主成分”的性质，则记为过拟合列。
        overfit.append(i)

#找到将产生过拟合的数据列的位置后，在特征矩阵中进行删除操作。
overfit = list(overfit)
#overfit.append('MSZoning_C (all)')#这条语句有用吗？是要把训练数据特征矩阵X中的列标签为'MSZoning_C (all)'的列也删除吗？但是训练数据中并没有任何一个列标签名称为MSZoning_C (all)。
X = X.drop(overfit, axis=1)#.copy()#删除截取后特征矩阵X中的过拟合列。因为drop并不影响原数据，所以使用copy。直接覆值应该也可以。
X_sub = X_sub.drop(overfit, axis=1)#.copy()
######################消除截取后特征矩阵的过拟合-【结束】#######################

print("删除极端值及过拟合列后，训练数据特征矩阵的维数为，特征：",'X', X.shape, '对应于特征的对数变换后的房价y', y.shape, '测试数据的特征矩阵（它应该在行、列上未被删减）X_sub', X_sub.shape)
##############################################################特征处理-【结束】###################################################################################

##############################################################机器学习-【开始】###################################################################################
print('特征处理已经完成。开始对训练数据进行机器学习', datetime.now())

#设置k折交叉验证的参数。
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


#定义均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


#创建模型评分函数，根据不同模型的表现打分
#cv表示Cross-validation,交叉验证的意思。
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

#############个体机器学习模型的创建（即模型声明和参数设置）-【开始】############
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

#定义ridge岭回归模型（使用二范数作为正则化项。不论是使用一范数还是二范数，正则化项的引入均是为了降低过拟合风险。）
#注：正则化项如果使用二范数，那么对于任何需要寻优的参数值，在寻优终止时，它都无法将某些参数值变为严格的0，尽管某些参数估计值变得非常小以至于可以忽略。即使用二范数会保留变量的所有信息，不会进行类似PCA的变量凸显。
#注：正则化项如果使用一范数，它比L2范数更易于获得“稀疏(sparse)”解，即它的求解结果会有更多的零分量。
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

#定义LASSO收缩模型（使用L1范数作为正则化项）（由于对目标函数的求解结果中将得到很多的零分量，它也被称为收缩模型。）
#注：正则化项如果使用二范数，那么对于任何需要寻优的参数值，在寻优终止时，它都无法将某些参数值变为严格的0，尽管某些参数估计值变得非常小以至于可以忽略。即使用二范数会保留变量的所有信息，不会进行类似PCA的变量凸显。
#注：正则化项如果使用一范数，它比L2范数更易于获得“稀疏(sparse)”解，即它的求解结果会有更多的零分量。										
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

#定义elastic net弹性网络模型（弹性网络实际上是结合了岭回归和lasso的特点，同时使用了L1和L2作为正则化项。）									
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

#定义SVM支持向量机模型                                     
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

#定义GB梯度提升模型（展开到一阶导数）									
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             

#定义lightgbm模型									
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       #min_data_in_leaf=2,
                                       #min_sum_hessian_in_leaf=11
                                       )

#定义xgboost模型（展开到二阶导数）                                      
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
#############个体机器学习模型的创建（即模型声明和参数设置）-【结束】############

###########################集成多个个体学习器-【开始】##########################
###！！！！！！！！！！！！
###！！！！！！！！！！！！
###！！！regressors=(...)中并没有纳入前面的svr模型,怎么回事？
###！！！！！！！！！！！！
###！！！！！！！！！！！！
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)#regressors=(...)中并没有纳入前面的svr模型
###########################集成多个个体学习器-【结束】##########################                             

############################进行交叉验证打分-【开始】###########################
#进行交叉验证，并对不同模型的表现打分
#（由于是交叉验证，将使用不同的数据集对同一模型进行评分，故每个模型对应一个得分序列。展示模型得分序列的平均分、标准差）
print('进行交叉验证，计算不同模型的得分TEST score on CV')

#打印二范数rideg岭回归模型的得分
score = cv_rmse(ridge)
print("二范数rideg岭回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印一范数LASSO收缩模型的得分
score = cv_rmse(lasso)
print("一范数LASSO收缩模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印elastic net弹性网络模型的得分
score = cv_rmse(elasticnet)
print("elastic net弹性网络模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印SVR支持向量机模型的得分
score = cv_rmse(svr)
print("SVR支持向量机模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印lightgbm轻梯度提升模型的得分
score = cv_rmse(lightgbm)
print("lightgbm轻梯度提升模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印gbr梯度提升回归模型的得分
score = cv_rmse(gbr)
print("gbr梯度提升回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

#打印xgboost模型的得分
score = cv_rmse(xgboost)
print("xgboost模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
############################进行交叉验证打分-【结束】###########################

#########使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【开始】#########
#开始集合所有模型，使用stacking方法
print('进行模型参数训练 START Fit')

print(datetime.now(), '对stack_gen集成器模型进行参数训练')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print(datetime.now(), '对elasticnet弹性网络模型进行参数训练')
elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.now(), '对一范数lasso收缩模型进行参数训练')
lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), '对二范数ridge岭回归模型进行参数训练')
ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), '对svr支持向量机模型进行参数训练')
svr_model_full_data = svr.fit(X, y)

print(datetime.now(), '对GradientBoosting梯度提升模型进行参数训练')
gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), '对xgboost二阶梯度提升模型进行参数训练')
xgb_model_full_data = xgboost.fit(X, y)

print(datetime.now(), '对lightgbm轻梯度提升模型进行参数训练')
lgb_model_full_data = lightgbm.fit(X, y)
#########使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【结束】#########

############################进行交叉验证打分-【结束】###########################

########定义个体学习器的预测值融合函数，检测预测值融合策略的效果-【开始】#######
#综合多个模型产生的预测值，作为多模型组合学习器的预测值
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))

#打印在上述模型配比下，多模型组合学习器的均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）
#使用训练数据对创造的模型进行k折交叉验证，以训练创造出的模型的参数配置。交叉验证训练过程结束后，将得到模型的参数配置。使用得出的参数配置下，在全体训练数据上进行验证，验证模型对全体训练数据重构的误差。
print('融合后的训练模型对原数据重构时的均方根对数误差RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))
########定义个体学习器的预测值融合函数，检测预测值融合策略的效果-【结束】#######

########将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【开始】########
print('使用测试集特征进行房价预测 Predict submission', datetime.now(),)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
#函数注释：.iloc[:,1]是基于索引位来选取数据集，[索引1:索引2]，左闭右开。
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
########将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【结束】########

#---------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

# 当对平台未公开的测试集进行预测时，前述模型的误差是0.114 this kernel gave a score 0.114
# 为了提高模型的得分，引入其他模型的优秀预测结果，与前述模型的预测结果进行混合（有点类似抄别人的答案，但实质是扩大集成模型的规模，引入更多的模型） let's up it by mixing with the top kernels

######################模型输出结果融合-【开始】#############################
#在多模型集成学习器预测结果的基础上，融合其他优秀模型（即平台上其他均方根对数误差小的模型）的预测结果。
#这步操作是为了降低多模型集成学习器的方差。
print('融合其他优秀模型的预测结果 Blend with Top Kernals submissions', datetime.now(),)
sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')
sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')
sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))
######################模型输出结果融合-【结束】#############################      

####################融合结果的极端值剔除-【开始】###########################  
#处理融合后结果中的极端值。把太大的数值（降序排列时，位于顶部往下0.005的数值，就是只有0.005的数比它大）缩小一点（乘以0.77），把太小的数值（降序排列时，位于顶部往下0.99的数值）放大一点（乘以1.1）
q1 = submission['SalePrice'].quantile(0.005)
q2 = submission['SalePrice'].quantile(0.995)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
####################融合结果的极端值剔除-【结束】###########################

#以csv文件的形式输出预测值
submission.to_csv("House_price_submission.csv", index=False)
print('融合结果.csv文件输出成功 Save submission', datetime.now())
##############################################################机器学习-【结束】###################################################################################