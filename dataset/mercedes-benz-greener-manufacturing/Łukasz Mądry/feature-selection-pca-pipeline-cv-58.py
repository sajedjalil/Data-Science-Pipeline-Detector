from sklearn.linear_model import RandomizedLasso, LinearRegression, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomTreesEmbedding
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, KFold, cross_val_predict
from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.drop('ID', axis=1, inplace=True)
test_id = test.pop('ID')

for k in train.columns:
    if np.unique(train[k]).shape[0] == 1:
        train.pop(k)
        test.pop(k)
#may be useful to test for some general variance thresholding instead of simply deleting constant columns

objs = train.select_dtypes(['object']).columns
ints = train.select_dtypes(['int']).columns.tolist()

y = train.pop('y')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

def pipe(stage_1, stage_2, y_1, y_2, to_pred, regr=RandomForestRegressor(max_depth=5, n_estimators=100)):
    
    #guess it should be totally doable on pipelines and unions but I'm lazy and don't want to write transforms
    
    selector_cat = SelectFromModel(RandomForestRegressor(max_depth=5, n_estimators=100), threshold='median')
    selector_bin = RandomizedLasso()
    
    selector_cat.fit(stage_1[objs], y_1)
    selector_bin.fit(stage_1[ints], y_1)
    
    trunc = np.hstack([selector_cat.transform(stage_2[objs]), selector_bin.transform(stage_2[ints])])
    trunc_test = np.hstack([selector_cat.transform(to_pred[objs]), selector_bin.transform(to_pred[ints])])
    
    pca = PCA(n_components=12, random_state=420).fit(stage_1)
    
    regr_no_pca = clone(regr)
    regr_with_pca = clone(regr)
    
    trunc_with_pca = np.hstack([trunc, pca.transform(stage_2)])
    trunc_with_pca_test = np.hstack([trunc_test, pca.transform(to_pred)])
    
    regr_no_pca.fit(trunc, y_2)
    regr_with_pca.fit(trunc_with_pca, y_2)
    
    return regr_no_pca.predict(trunc_test)/2 + regr_with_pca.predict(trunc_with_pca_test)/2
    
x_1, x_2, y_1, y_2 = train_test_split(train, y, train_size=.5)

preds = [pipe(x_1, x_2, y_1, y_2, test)]
preds = np.vstack(preds).mean(axis=0)


pd.DataFrame({'ID': test_id, 'y': preds}).to_csv('sub_avg.csv', index=False)