'''
Data goes through pipeline that includes data cleaning, and feature processing. Initially data is loaded as pandas
dataframe but during processing in the pipeline some steps return sparse matrices, to store them separately data format
is changed from dataframe to dictionary that stores initial dataframe and sparse data.
Text columns (item_description, name) are encoded with tfidfVectorizer, brand_name and item_category are considered
categorical and are one hot encoded (item_category has several values for each item and class OneHotMultipleEncoder
handles that), item_condition(values in range (1,5)) is scaled to be in range (0,1). Since data is heterogeneous
and differenrt transforms are applied to different featuers, sklearn classes' OneHotEncoder, MinMaxScaler,
TfidfVectorizer fit/transform methods are overridden to act only on selected features. Best estimator, its
hyperparameters, transformers' parameters are determined by grid search
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, coo_matrix
from sklearn.linear_model import SGDRegressor
import unicodedata
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

class Clean:

    def transform(self, X):
        return Clean.clean(X)

    def fit(self, X, params=None):
        return self

    @staticmethod
    def clean(data):
        '''
        fills na and processes text columns
        :param data: DataFrame, feature matrix
        :return: DataFrame, modified feature matrix
        '''
        data = data.copy()
        data.fillna('nan', inplace=True)
        text_cols = [col for col in data if data[col].dtype == object]
        for text_col in text_cols:
            data.loc[:, text_col] = data.loc[:,text_col].str.lower()
            data.loc[:, text_col] = data.loc[:, text_col].str.replace('\xae', '')
            data.loc[:,text_col] = data.loc[:,text_col].map(lambda x: Clean.remove_accents(x))
        data.loc[:,'name'] = data.loc[:,'name'].str.replace(r'[rm]', '')
        data.loc[:,'item_description'] = data.loc[:,'item_description'].str.replace(r'[rm]', '')
        return data

    @staticmethod
    def remove_accents(input_str):
        '''
        removes accents
        :param input_str: str, input
        :return: str, input_str without accents
        '''
        #print(input_str)
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


class OneHotMultipleEncoder:
    '''
    class to encode categories to one hot, but each item may have several categories at the same time
    '''
    def __init__(self, column):
        '''
        :param column: str, name of the DataFrame column to process
        :param dict_categories: dict, {category_name:category_id}
        '''
        self.column = column

    def fit(self, X, params=None):
        '''
        computes dict_categories {category_name:category_id}
        :param X: dict, {'dense':DataFrame, 'sparse':scipy.sparse array}
        :param params: ignored
        :return: self
        '''
        self.dict_categories = OneHotMultipleEncoder.get_values_dict(X['dense'], self.column)
        return self

    def transform(self, X):
        '''
        deletes processed column from dataframe and stacks a sparse matrix of learned features to the sparse part
        :param X: dict, {'dense':DataFrame, 'sparse':scipy.sparse array}
        :return: X: dict, {'dense':DataFrame, 'sparse':scipy.sparse array}, transformed data
        '''
        n = len(X['dense'])
        indexes = ([], [])
        index = zip(range(n), X['dense'].index)
        for i, j in index:
            category = X['dense'].loc[j, self.column].split('/')
            if X['dense'].loc[j, self.column] == 'nan':
                continue
            for cat in category:
                if cat in self.dict_categories.keys():
                    indexes[0].append(i)
                    indexes[1].append(self.dict_categories[cat])
        values = np.ones((len(indexes[0])))
        X['sparse'] = coo_matrix((values, indexes), shape=((n, len(self.dict_categories))))
        del X['dense'][self.column]
        return X

    @staticmethod
    def get_values_dict(data, column):
        '''
        creates a dictionary of ordinal encoding of a categorical feature
        :param data: DataFrame, feature matrix
        :param column: str, name of the feature to encode
        :return: dict, key - str, name of the category, value - int, encoding
        '''
        n = len(data)
        set_of_cat = set()
        for i in data.index:
            category = data.loc[i, column].split('/')
            if data.loc[i, column] == 'nan':
                continue
            for cat in category:
                set_of_cat.add(cat)
        return {name: id for name, id in zip(list(set_of_cat), range(len(set_of_cat)))}


class ToDict:
    '''
    transforms dataframe to a dict (to keep dense and sparse parts of the feature matrix separate)
    '''

    def fit(self, X, params=None):
        return self

    def transform(self, X, params=None):
        return {'dense': X, 'sparse': True}


class ToArray:
    '''
    converts dense feature matrix to sparse and stacks them together
    '''

    def fit(self, X, params=None):
        return self

    def transform(self, X):
        ans = X['sparse']
        for name in X['dense']:
            m2 = coo_matrix(np.array(X['dense'].loc[:,name]).reshape(ans.shape[0], -1))
            ans = hstack((ans, m2))
        return ans


class TfidfVectorizer(TfidfVectorizer):
    '''
    sklearn TfidfVectorizer with modified methods
    - uses only one column from the passed data frame
    - writes sparse results to the sparse part of the data
    '''
    def __init__(self, column, **params):
        self.column = column
        super().__init__(params)

    def fit(self, X, **params):
        return super().fit(X.loc[:,self.column].fillna('No description yet'), params)

    def transform(self, X):
        data = X['dense'].loc[:,self.column].fillna('No description yet')
        ans = super().transform(data)
        if type(X['sparse'])==bool:
            X['sparse'] = ans
        else:
            X['sparse'] = hstack([X['sparse'], ans])
        del X['dense'][self.column]
        return X

    def fit_transform(self, X, params = None):
        self.fit(X['dense'])
        return self.transform(X)

    def set_params(self, **params):
        for key,value in params.items():
            self.key = value
        return self


class OneHotEncoder(OneHotEncoder):
    '''
    sklearn OneHotEncoder with modified methods (uses only one column from the passed data frame)
    '''
    def __init__(self, column, params=None):
        self.column = column
        super().__init__(sparse=False, handle_unknown = 'ignore')

    def fit(self, X, params=None):
        data = np.array(X.loc[:,self.column].fillna('no brand', inplace=True), dtype='S100').reshape(-1, 1)
        return super().fit(data)

    def transform(self, X):
        data = X.loc[:,self.column]
        ans = pd.DataFrame(super().transform(np.array(data).reshape(-1, 1)), index = X.index)
        X = pd.concat([X, ans], axis = 1)
        del X[self.column]
        return X


class MinMaxScaler(MinMaxScaler):
    '''
    sklearn MinMaxScaler with modified methods (uses only one column from the passed data frame)
    '''
    def __init__(self, column, params=None):
        self.column = column
        self.feature_range = (0, 1)
        self.copy = True
        super()

    def fit(self, X, params=None):
        return super().fit(np.array(X.loc[:,self.column], dtype=float).reshape(-1, 1))

    def transform(self, X):
        X.loc[:,self.column] = super().transform(np.array(X.loc[:,self.column], dtype=float).reshape(-1, 1))
        return X


def rmsle(y, y_p) :
    '''
    root mean squared log error
    :param y: array-like, ground truth
    :param y_p: array-like, predictions
    :return: float, score
    '''
    assert len(y) == len(y_p)
    y_p = np.array([max(y,0) for y in y_p])
    score = np.sqrt(np.mean((np.log(1+y_p) - np.log(1+y))**2))
    if score <= 0:
        print(y_p)
    return score




def submit(prediction, filename = 'submit.csv'):
    '''
    writes predictions to a csv file
    :param prediction: array-like, predictions
    :param filename: string, name of the file, default 'submit.csv'
    :return: DataFrame, modified feature matrix
    '''
    Id = range(len(prediction))
    d = {'test_id': Id, 'price': prediction}
    submission = pd.DataFrame(data = d)
    submission.to_csv(filename, index = False)

n = 10000
data_train = pd.read_table("../input/train.tsv", index_col='train_id')
data_train = data_train.sample(n)
y = data_train['price']
del data_train['price']
pipe = Pipeline(steps = [('cleaning', Clean()),
                         ('brand_encoding', OneHotEncoder('brand_name')),
                         ('item_condition_scaling', MinMaxScaler('item_condition_id')),
                         ('change_format', ToDict()),
                         ('category_encoding', OneHotMultipleEncoder('category_name')),
                         ('item_description_tfidf',TfidfVectorizer('item_description', lower_case = False, stop_words='english')),
                         ('name_tfidf',TfidfVectorizer('name')),
                         ('hstack_arrays', ToArray()),
                         ('imputer', SimpleImputer(strategy = 'median')),
                         ('estimator', SVR())])
param_grid = [
              {
                'estimator' : [SVR()],
                'estimator__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                'estimator__degree' : list(range(3,6)),
                'estimator__C' : [10**x for x in range(-3,4,1)],
                'estimator__gamma': ['scale'] + [10**x for x in range(-3,4,1)],
                'item_description_tfidf__nrgam_range' : list(zip([1]*3, list(range(1,4)))),
                'item_description_tfidf__max_features': [None]+list(range(100, 1000, 100))
              },
              {
                 'estimator': [SGDRegressor(learning_rate = 'invscaling', penalty = 'elasticnet')],
                 'item_description_tfidf__nrgam_range' : list(zip([1]*3, list(range(1,4)))),
                 'item_description_tfidf__max_features': [None]+list(range(100, 1000, 100))
              },
              ]
#pipe.fit(data_train[:10], y[:10])
#pipe.predict(data_train)
#print(rmsle(pipe, data_train, y))
#for param_grid in param_grids:
    # print(param_grid)
    # grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid, scoring = rmsle, cv = 3)
    # grid_search.fit(data_train, y)
    # best_params = grid_search.best_params_
    # print(grid_search.cv_results_)
    # print(grid_search.best_score_)
    # print(best_params)
# pipe.set_params(best_params)
# data_test = pd.read_table("../input/test.tsv")
# submit(pipe.predict(data_test))
scoring = make_scorer(rmsle, greater_is_better = False)
grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid, scoring = scoring, cv = 3)
#print((pipe.predict(data_train)<=0).any())

grid_search.fit(data_train, y)
best_params = grid_search.best_params_
with open('log2.txt', 'w') as f:
    print(grid_search.cv_results_, file = f)
print(grid_search.best_score_)
print(best_params)