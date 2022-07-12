# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import sklearn
import scipy

# Portions of this code is from https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        print(c)
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

#print(train)
#sys.exit()

usable_columns = list(set(train.columns) - set(['y','ID','X325'
,'X13'
,'X164'
,'X358'
,'X155'
,'X211'
,'X221'
,'X251'
,'X362'
,'X74'
,'X190'
,'X194'
,'X185'
,'X99'
,'X124'
,'X217'
,'X287'
,'X38'
,'X294'
,'X83'
,'X180'
,'X166'
,'X129'
,'X31'
,'X32'
,'X174'
,'X355'
,'X235'
,'X14'
,'X21'
,'X296'
,'X356'
,'X26'
,'X212'
,'X282'
,'X181'
,'X250'
,'X372'
,'X334'
,'X197'
,'X136'
,'X249'
,'X337'
,'X302'
,'X328'
,'X299'
,'X20'
,'X122'
,'X97'
,'X157'
,'X307'
,'X347'
,'X135'
,'X385'
,'X313'
,'X377'
,'X175'
,'X77'
,'X66'
,'X349'
,'X53'
,'X202'
,'X230'
,'X182'
,'X214'
,'X331'
,'X69'
,'X45'
,'X109'
,'X168'
,'X62'
,'X318'
,'X219'
,'X37'
,'X183'
,'X272'
,'X94'
,'X268'
,'X285'
,'X357'
,'X42'
,'X384'
,'X281'
,'X253'
,'X101'
,'X359'
,'X39'
,'X106'
,'X260'
,'X123'
,'X336'
,'X98'
,'X88'
,'X56'
,'X259'
,'X308'
,'X79'
,'X110'
,'X108'
,'X305'
,'X57'
,'X41'
,'X92'
,'X90'
,'X323'
,'X289'
,'X295'
,'X176'
,'X278'
,'X311'
,'X67'
,'X352'
,'X65'
,'X317'
,'X44'
,'X254'
,'X138'
,'X304'
,'X50'
,'X269'
,'X192'
,'X102'
,'X126'
,'X89'
,'X376'
,'X143'
,'X252'
,'X320'
,'X198'
,'X277'
,'X246'
,'X24'
,'X280'
,'X239'
,'X247'
,'X60'
,'X368'
,'X34'
,'X319'
,'X55'
,'X248'
,'X367'
,'X208'
,'X146'
,'X125'
,'X167'
,'X380'
,'X258'
,'X81'
,'X159'
,'X172'
,'X27'
,'X298'
,'X350'
,'X112'
,'X187'
,'X153'
,'X370'
,'X338'
,'X309'
,'X233'
,'X297'
,'X141'
,'X33'
,'X76'
,'X87'
,'X288'
,'X242'
,'X137'
,'X290'
,'X59'
,'X231'
,'X213'
,'X15'
,'X373'
,'X256'
,'X85'
,'X210'
,'X379'
,'X11'
,'X73'
,'X216'
,'X133'
,'X360'
,'X107'
,'X369'
,'X209'
,'X195'
,'X265'
,'X145'
,'X274'
,'X199'
,'X333'
,'X207'
,'X375'
,'X116'
,'X361'
,'X346'
,'X171'
,'X36'
,'X40'
,'X366'
,'X255'
,'X150'
,'X18'
,'X301'
,'X270'
,'X353'
,'X371'
,'X93'
,'X63'
,'X28'
,'X382'
,'X330'
,'X140'
,'X343'
,'X80'
,'X169'
,'X241'
,'X186'
,'X68'
,'X10'
,'X310'
,'X293'
,'X243'
,'X257'
,'X327'
,'X335'
,'X200'
,'X160'
,'X17'
,'X332'
,'X245'
,'X191'
,'X139'
,'X156'
,'X228'
,'X344'
,'X223'
,'X227'
,'X374'
,'X284'
,'X348'
,'X154'
,'X22'
,'X78'
,'X165'
,'X262'
,'X238'
,'X128'
,'X49'
,'X144'
,'X117'
,'X179'
,'X161'
,'X276'
,'X220'
,'X234'
,'X273'
,'X354'
,'X114'
,'X35'
,'X91'
,'X286'
,'X111'
,'X46'
,'X82'
,'X103'
,'X130'
,'X162'
,'X184'
,'X120'
,'X237'
,'X177'
,'X96'
,'X340'
,'X30'
,'X43'
,'X58'
,'X378'
,'X300'
,'X64'
,'X225'
,'X363'
,'X196'
,'X342'
,'X322'
,'X364'
,'X100'
,'X151'
,'X152'
,'X224'
,'X240'
,'X19'
,'X75'
,'X203'
,'X324'
,'X70'
,'X329'
,'X206'
,'X226'
,'X291'
,'X52'
,'X201'
,'X163'
,'X215'
,'X345'
,'X12'
,'X178'
,'X4'
,'X16'
,'X148'
,'X173'
,'X218'
,'X189'
,'X264'
,'X86'
,'X132'
,'X84'
,'X365'
,'X115'
,'X229'
,'X170'
,'X292'
,'X3'
,'X312'
,'X321'
,'X341'
,'X71'
,'X326'
,'X244'
,'X266'
,'X351'
,'X142']))

train['add1']=train['X316']^train['X283']
test['add1']=test['X316']^test['X283']
usable_columns.append('add1')
train['add2']=train['X283']^train['X271']
test['add2']=test['X283']^test['X271']
usable_columns.append('add2')
#train['add3']=train['X275']^train['X158']
#test['add3']=test['X275']^test['X158']
#usable_columns.append('add3')

print(usable_columns)
print(len(usable_columns))
np.random.shuffle(usable_columns)
usable_columns=usable_columns[0:376]

for c in usable_columns:
    print("test['%s']+"%c, end='')
#usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
#y_train=y_train-np.min(y_train)
#y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

r2train=[]
r2val=[]
kf = KFold(n_splits=5,shuffle=True,random_state=33)
n=0
for train_index, test_index in kf.split(finaltrainset):
    model=GradientBoostingRegressor(n_estimators=50,max_depth=2,random_state=33)
    #model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
    model.fit(finaltrainset[train_index], y_train[train_index])

    r2train.append(r2_score(y_train[train_index],model.predict(finaltrainset[train_index])))
    r2val.append(r2_score(y_train[test_index],model.predict(finaltrainset[test_index])))
    print(r2train[-1],r2val[-1])
   
    if n==0: 
        y_pred = model.predict(finaltestset)#+np.min(y_train)
        fi=model.feature_importances_
    else: 
        y_pred += model.predict(finaltestset)#+np.min(y_train)
        fi+=model.feature_importances_
    
    n+=1

print('******')
for i in range(0,len(fi)):
    print(fi[i],usable_columns[i])
for i in range(0,len(fi)):
    #print(fi[i],usable_columns[i])
    if fi[i]==0: print(",'%s'"%usable_columns[i])

print('******')
print(np.mean(r2train))
print(np.mean(r2val))
#sys.exit()

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred/n
sub.to_csv('stacked-models.csv', index=False)
