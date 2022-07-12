global ITEM_ID 
#ITEM_ID = 2
N_PERIODS = 91
#STORE_ID = 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
%matplotlib inline
plt.rcParams['figure.figsize'] = 18,6


class RollingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,  rolling_size, function):
        self.rolling_size = rolling_size
        self.function  = function
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        selection = [x for x in X.columns if 'shift' in x][0]
        X.loc[:, f'roll_{self.rolling_size}_{self.function.__name__}']  = \
        X[selection].rolling(self.rolling_size).apply(lambda x: self.function(x))
        #X.dropna(axis=1, how='all', inplace = True)
        return X


class ShiftTransformer(BaseEstimator, TransformerMixin, object):
    def __init__(self, periods=90):
        self.periods = periods
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ShiftTransformer, cls).__new__(cls)
        return cls.instance
    
    def _fit(self, X, y=None):
        periods = self.periods
        ser = pd.Series(index = pd.date_range(start=pd.date_range(start = y.index[0], freq = "D", periods=periods)[-1],
                                              end=pd.date_range(start = y.index[-1], freq = "D", periods=periods)[-1], 
                                              freq = "D"),
                        data=y.values)
        self.shifted_data = ser
        return self
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.loc[:, f'shift_{self.periods}']  = self.shifted_data
        X.dropna(axis=1, how='all', inplace = True)
        return X

class SeasonalTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        seasonalities = {}
        ser = y
        decomp_results = seasonal_decompose(ser, freq=7).seasonal
        decomp_results = decomp_results.to_frame().reset_index()
        decomp_results['weekday'] = decomp_results['date'].dt.weekday
        temp = decomp_results.set_index('weekday')
        temp.columns = ['date', 'seasonal']
        seasonalities = temp.to_dict()
        self.seasonalities = seasonalities
        return self
    
    def transform(self, X, y=None):
        X.reset_index(inplace = True)
        X['weekday'] = X['date'].dt.weekday
        X['seasonal_value'] = X['weekday'].map(self.seasonalities)
        X.set_index('date', inplace = True)
        return X

class DatesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.reset_index(inplace = True)
        X['month'] = X['date'].dt.month
        X['week'] = X['date'].dt.week
        X.set_index('date', inplace = True)
        return X
    
train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'], index_col='date')
test['sales'] = np.nan
df = train.groupby(['date','item'])['sales'].sum().unstack()

for ITEM_ID in range(1,51):
    y = df[ITEM_ID]
    X = pd.DataFrame(index = y.index)

    tss = TimeSeriesSplit()
    #Estimator = LinearRegression()
    #X = X.dropna()
    #y = y[X.index]
    Estimator = LGBMRegressor()

    scores = []
    sh = ShiftTransformer(periods = N_PERIODS)._fit(X,y) #1.add .fit(X,y)

    for train_ix, test_ix in tss.split(X):
        X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        ft = FeatureUnion([('dates', DatesTransformer()),
                           ('seas', SeasonalTransformer()),
                           ('shift', ShiftTransformer()),
                           #('rw', RollingWindowTransformer(14, np.std))
                          ])
        est = Estimator
        ft.fit_transform(X_train, y_train)

        X_train = X_train.dropna()
        y_train = y_train[X_train.index]

        est.fit(X_train, y_train)
        ft.transform(X_test)
        scores.append(est.score(X_test, y_test)) 

    X_predict = pd.DataFrame(index = test.index.unique())
    X_predict.index.name = 'date'
    ft.transform(X_predict)
    y_pred = pd.Series(est.predict(X_predict), index = X_predict.index)

    gr = train.groupby(['store','date','item'])['sales'].sum().unstack('item')
    store = gr[ITEM_ID].unstack('store')
    storeshare_mean = store.divide(store.sum(axis=1), axis=0).mean().to_dict()
    storeshare_std = {}
    for i in range(1,11):
        q = store.divide(store.sum(axis=1), axis=0)[i]
        storeshare_std[i] = np.std(q-q.mean())

    for STORE_ID in range(1,11):
        #test['sales'][(test.item==ITEM_ID)&(test.store == STORE_ID)] = y_pred * storeshare_mean[STORE_ID] #без добавления шума
        test['sales'][(test.item==ITEM_ID)&(test.store == STORE_ID)] = \
        y_pred * (storeshare_mean[STORE_ID] + np.random.normal(loc=0, scale = storeshare_std[STORE_ID], size = len(y_pred))) #c добавлением шума

    print(len(test['sales'].dropna()))

test.set_index('id')['sales'].to_csv('submission.csv')