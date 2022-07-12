import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from joblib import dump, load
from statistics import mean
import numpy as np
from bayes_opt import BayesianOptimization
from collections import Counter
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


class dataset:

    def summary(self):
        pd.set_option('display.max_columns', 500)
        print(self.data_frame.head())
        print(self.data_frame.describe())
        print(self.data_frame.dtypes)
        missing = self.data_frame.isna().sum()
        print('Missing data:' + missing) if sum(missing) > 0 else print('No missing data.')
        features = [f for f in self.data_frame.columns.values if self.data_frame[f].dtype != 'O']
        sb.heatmap(self.data_frame[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")

    def load_frame(self, file_path, index_col_name = None):
        self.data_frame = pd.read_csv(file_path, index_col = index_col_name)
        print(self.data_frame.head())

    def features(self, features, labels):
        self.features = features
        self.labels = labels

    def drop(self, *columns_to_drop):

        self.data_frame.drop(labels =[col for col in columns_to_drop] , axis = 1, inplace = True)
        print(self.data_frame.head())

    def log(self, *columns_to_log):

        for col in columns_to_log:
            self.data_frame = self.data_frame[col].map(lambda i: np.log(i) if i > 0 else 0)

    def dummify(self):

        self.data_frame = pd.get_dummies(self.data_frame)

    def detect_outliers(self, ignore = None, n=2):

        features = []

        for feature, type in self.data_frame.dtypes.to_dict().items():
            if type != 'O': features.append(feature)

        if ignore:
            features.remove(ignore)

        print(features)


        outlier_indices = []

        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(self.data_frame[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(self.data_frame[col], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = self.data_frame[
                (self.data_frame[col] < Q1 - outlier_step) | (self.data_frame[col] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        outlier_count = len(multiple_outliers)
        outlier_proportion = int((outlier_count/len(self.data_frame))*100)

        print(f'{outlier_count} outliers detected. {outlier_proportion}% of total')

        self.outliers = multiple_outliers

    def drop_outliers(self):

        self.detect_outliers()
        self.data_frame.drop(self.outliers, axis=0, inplace=True)

    def count_missing(self):
        print(self.data_frame.isna().sum())

    # def impute_missing(self, *features):
        # for feature in features:

    def barplot(self, x, y):
        sb.factorplot(x=x, y=y, data=self.data_frame, kind="bar", size=6,
                       palette="muted")

    def split(self, val_proportion = None, test_proportion = None, shuffle = False):

        if shuffle:
            self.data_frame = self.data_frame.sample(frac=1)

        if test_proportion:
            self.test_length = int(len(self.data_frame)*test_proportion)
            print(f'Test sample size of {len(self.test_length)}')

        if val_proportion:
            val_length = int(len(self.data_frame) * val_proportion)
            print(f'Validation sample size of {len(self.val_length)}')

        self.train_length = len(self.data_frame) - (self.test_length + self.val_length)

        print(f'Training sample size of {self.train_length}')

class model_foundation():

    def __init__(self, multiclass=True):
        self.multiclass = multiclass
        self.convert = True

    def fit(self, x, y):
        raise NotImplementedError

    def parameters(self, **params):
        for param, value in params:
            setattr(self, param, value)

    def predict(self, x):
        predictions = self.model.predict(x)
        if self.multiclass and self.convert:
            predictions = self.eval_qwk_lgb_regr(predictions)
        else:
            predictions = np.array(list(predictions))
        return predictions

    def k_fold_validation(self, x, y, nfolds, train_on_all = False):
        self.true = y
        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        oof_pred = np.zeros((len(x)))
        train_acc = []
        for fold, (tr_ind, val_ind) in enumerate(kf.split(x, y)):
            print('Fold {}'.format(fold + 1))
            x_train, x_val = x[tr_ind], x[val_ind]
            y_train, y_val = y[tr_ind], y[val_ind]
            model = self.fit(x_train, y_train, x_val = x_val, y_val = y_val)
            oof_pred[val_ind] = self.predict(x_val)
            if self.convert:
                train_pred = self.predict(x_train)
                train_acc.append(int((100 * sum(train_pred == y_train)) / len(train_pred)))
        if self.multiclass:
            oof_pred = oof_pred
        if train_on_all:
            model = self.fit(x,y)
        if self.convert:
            loss_score = cohen_kappa_score(y, oof_pred, weights='quadratic')
            self.kappa = loss_score
            val_acc = int((100 * sum(oof_pred == y) )/ len(oof_pred))
            print(f'kappa {loss_score}')
            print(f'train acc {mean(train_acc)}%')
            print(f'val acc {val_acc}%')
            self.train_acc = mean(train_acc)
            self.val_acc = val_acc
        self.oof_pred = oof_pred



    def eval_qwk_lgb_regr(self, y_pred):
        """
        Fast cappa eval function for lgb.
        """
        dist = Counter(self.true)
        for k in dist:
            dist[k] /= len(self.true)
        acum = 0
        bound = {}
        for i in range(3):
            acum += dist[i]
            bound[i] = np.percentile(y_pred, acum * 100)

        def classify(x):
            if x <= bound[0]:
                return 0
            elif x <= bound[1]:
                return 1
            elif x <= bound[2]:
                return 2
            else:
                return 3

        return np.array(list(map(classify, y_pred)))



class lgb_model(model_foundation):
    
    def __init__(self, multiclass = True):
        self.params = {'n_estimators':5000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,
                    'lambda_l2': 1
                    }
        self.multiclass = multiclass
        self.model_type = 'lgb'
        self.convert = True

    def fit(self, x_train, y_train, x_val = False, y_val = False):
        train_set = lgb.Dataset(x_train, y_train)
        if type(x_val) == bool:
            self.model = lgb.train(self.params, train_set)
            return
        val_set = lgb.Dataset(x_val, y_val)
        self.model = lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)

class xgb_model(model_foundation):

    def __init__(self, multiclass = True):
        self.params = {'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'max_depth': 10,
            'subsample': 1,
            'objective':'reg:squarederror',
            #'eval_metric':'rmse',
            'min_child_weight':3,
            'gamma':0.25,
            'n_estimators':5000}
        self.multiclass = multiclass
        self.model_type = 'xgb'
        self.convert = True

    def init(self):
        self.weight = 'balanced'
        self.depth = 10
        self.estimators = 100

    def predict(self, x):
        x = xgb.DMatrix(x)
        predictions = self.model.predict(x)
        if self.multiclass and self.convert:
            predictions = self.eval_qwk_lgb_regr(predictions)
        else:
            predictions = np.array(list(predictions))
        return predictions

    def fit(self, x_train, y_train, x_val = False, y_val = False):
        train_set = xgb.DMatrix(x_train, y_train)
        if type(x_val) == bool:
            self.model = xgb.train(self.params, train_set)
            return
        val_set = xgb.DMatrix(x_val, y_val)
        self.model = xgb.train(self.params, train_set, num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], verbose_eval=100, early_stopping_rounds=100)

class nn_model(model_foundation):

    def __init__(self, multiclass = True):
        self.model_type = 'nn'
        self.multiclass = multiclass
        self.convert = True
        self.layers = {'n_layers': 5, 'layers':{1: 200, 2:100, 3:50, 4:25, 5:1}, 'dropout':0.3}
    def fit(self, x_train, y_train, x_val = False, y_val = False):
        data = list(x_train.copy())
        if not type(x_val) == bool:
            data.extend(list(x_val.copy()))
        scaler = MinMaxScaler()
        scaler.fit(data)
        self.scaler = scaler
        del data
        x_train = scaler.transform(x_train)
        if not type(x_val) == bool:
            x_val = scaler.transform(x_val)
        model = tf.keras.models.Sequential([
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(25, activation='relu'),
            keras.layers.LayerNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='relu')
        ])
        # for layer in self.layers['layers']:
        #     model.add(keras.layers.Dense(self.layers['layers'][layer], activation = 'relu'))
        #     if layer != len(self.layers['layers']):
        #         model.add(keras.layers.LayerNormalization(), keras.layers.Dropout(self.layers['dropout']))


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse')
        print(model.summary())
        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True,
                                                       verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        if not type(x_val) == bool:
            model.fit(x_train,
                      y_train,
                      validation_data=(x_val, y_val),
                      epochs=100,
                      callbacks=[save_best, early_stop])
        else:
            model.fit(x_train,
                      y_train,
                      epochs=100,
                      callbacks=[save_best, early_stop])
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)
        predictions = self.model.predict(x)
        if self.multiclass and self.convert:
            predictions = self.eval_qwk_lgb_regr(predictions)
        else:
            predictions = np.array(list(predictions))[:,0]
        return predictions

class ensemble:
    def __init__(self):
        self.models = {}

    def add_model(self, model_class, weight = 0):

        name = model_class.model_type
        self.models[name] = {}
        self.models[name]['model'] = model_class
        self.models[name]['weight'] = weight

    def predict(self, x):

        predictions = [0]*len(x)
        for model in self.models:
            preds = list(self.models[model]['model'].predict(x)*self.models[model]['weight'])
            predictions = [sum(x) for x in zip(preds, predictions)]
        return predictions

    def kappa(self):

        first = list(self.models.keys())[0]
        predictions = [0] * len(self.models[first]['model'].oof_pred)
        for model in list(self.models.keys()):
            preds = self.models[model]['model'].oof_pred*self.models[model]['weight']
            predictions = [sum(x) for x in zip(preds, predictions)]

        predictions = self.models[first]['model'].eval_qwk_lgb_regr(predictions)
        kappa = cohen_kappa_score(self.models[first]['model'].true, predictions, weights='quadratic')
        return kappa

def get_data(test = False, path = 'train.csv'):
    d = dataset()
    d.load_frame(file_path = path)
    d.data_frame.drop(labels = ['event_count'] , axis = 1, inplace = True)
    d.data_frame = d.data_frame.sort_values(by = ['installation_id', 'timestamp'])
    d.data_frame['attempt'] = ((d.data_frame['event_data'].str.contains('4100}') & (d.data_frame['title'].str.contains('Bird Measurer') == False)) | (d.data_frame['event_data'].str.contains('4110}') & d.data_frame['title'].str.contains('Bird Measurer'))) & (d.data_frame['type'].str.contains('Assessment')).values
    d.data_frame = d.data_frame.assign(success = (d.data_frame['event_data'].str.contains('correct":true') & d.data_frame['attempt']).astype(int))
    d.data_frame.drop(labels = ['event_data'] , axis = 1, inplace = True)

    first = True
    count = 1
    while True:

        if test:
            ids = list(dict.fromkeys(d.data_frame['installation_id']))
            tdata = pd.DataFrame(index = ids)
        else:
            ids = list(dict.fromkeys(d.data_frame[d.data_frame['attempt'].values]['installation_id']))
            if len(ids) == 0:
                break
            d.data_frame = d.data_frame[d.data_frame['installation_id'].isin(ids)]
            no_attempts = d.data_frame[d.data_frame['type'] == 'Assessment'].groupby(['game_session'], sort=False)['attempt'].sum() == 0
            d.data_frame = d.data_frame[d.data_frame['game_session'].isin(no_attempts[no_attempts].index.values) == False]

            final_attempts = d.data_frame.groupby(['installation_id'], sort = False).apply(lambda g: g[(g['game_session']==g[g['type'] == 'Assessment'].tail(1)['game_session'].values[0])])
            correct_attempts_final = final_attempts.reset_index(drop=True).groupby(['installation_id', 'game_session', 'title'], sort=False)['success'].sum()
            total_attempts_final = final_attempts.reset_index(drop=True).groupby(['installation_id'], sort=False)['attempt'].sum()

            preds = []
            for i in range(0,len(correct_attempts_final)):
                if total_attempts_final[i] == 0 or correct_attempts_final[i] == 0:
                    preds.append(0)
                else:
                    ratio = correct_attempts_final[i]/total_attempts_final[i]
                    if ratio == 1:
                        preds.append(3)
                    elif ratio == 0.5:
                        preds.append(2)
                    else:
                        preds.append(1)
            tdata = pd.DataFrame({'installation_id': correct_attempts_final.reset_index()['installation_id'], 'accuracy_group': preds})
            tdata.set_index('installation_id', inplace = True)

        #Truncate

            d.data_frame = d.data_frame.groupby(['installation_id'], sort = False).apply(lambda g: g[g['timestamp']<=g[g['game_session']==g[g['type'] == 'Assessment'].tail(1)['game_session'].values[0]]['timestamp'].values[0]])
            d.data_frame = d.data_frame.reset_index(level=0, drop=True).reset_index(drop=True)
        #Group
        #Create training frame with features
        tdata = tdata.assign(total_time = list(pd.DataFrame(d.data_frame.groupby(['installation_id', 'game_session'], sort=False)['game_time'].max()).groupby(['installation_id'], sort = False).sum()['game_time']))
        tdata = tdata.assign(mean_time = list(pd.DataFrame(d.data_frame.groupby(['installation_id', 'game_session'], sort=False)['game_time'].max()).groupby(['installation_id'], sort = False).mean()['game_time']))
        tdata = tdata.assign(std_time = list(pd.DataFrame(d.data_frame.groupby(['installation_id', 'game_session'], sort=False)['game_time'].max()).groupby(['installation_id'], sort = False).std()['game_time']))
        tdata = tdata.assign(total_time_assessments=list(pd.DataFrame(d.data_frame[d.data_frame['type'] == 'Assessment'].groupby(['installation_id', 'game_session'], sort=False)['game_time'].max()).groupby(['installation_id'], sort=False).sum()['game_time']))
        tdata = tdata.assign(mean_time_assessments=list(pd.DataFrame(d.data_frame[d.data_frame['type'] == 'Assessment'].groupby(['installation_id', 'game_session'], sort=False)['game_time'].max()).groupby(['installation_id'], sort=False).mean()['game_time']))
        tdata = tdata.assign(std_time_assessments=list(pd.DataFrame(d.data_frame[d.data_frame['type'] == 'Assessment'].groupby(['installation_id', 'game_session'], sort=False)['game_time'].max()).groupby(['installation_id'], sort=False).std()['game_time']))
        tdata = tdata.assign(title_count = list(d.data_frame.groupby(['installation_id'], sort = False)['title'].nunique()))
        #tdata = tdata.assign(activity_count = list(d.data_frame.groupby(['installation_id'], sort = False)['timestamp'].nunique()))
        #tdata = tdata.assign(event_code_count = list(d.data_frame.groupby(['installation_id'], sort = False)['event_code'].nunique()))
        tdata = tdata.assign(event_id_count = list(d.data_frame.groupby(['installation_id'], sort = False)['event_id'].nunique()))
        #d.data_frame['title_event_code'] = d.data_frame['event_id']+d.data_frame['title']
        #tdata = tdata.assign(title_event_code = list(d.data_frame.groupby(['installation_id'], sort = False)['title_event_code'].nunique()))
        tdata = tdata.assign(assessment = list(d.data_frame[d.data_frame['type'] == 'Assessment'].groupby(['installation_id'], sort = False).tail(1)['title'] + ' final assessment'))
        tdata = tdata.assign(session_count = list(d.data_frame.groupby(['installation_id'], sort = False)['game_session'].nunique()))

        tdata = pd.get_dummies(tdata)


        correct_attempts = d.data_frame[d.data_frame['attempt'].values].groupby(['installation_id', 'game_session', 'title'], sort=False)['success'].sum()
        total_attempts = d.data_frame[d.data_frame['attempt'].values].groupby(['installation_id', 'game_session', 'title'], sort=False)['success'].count()
        average_accuracies = correct_attempts/total_attempts
        assessment = d.data_frame[d.data_frame['attempt'].values].groupby(['installation_id', 'game_session', 'title'], sort=False).tail(1)
        all_accuracy_groups = (average_accuracies==1)*3+(average_accuracies==0.5)*2+(average_accuracies<0.5)*1
        tdata = tdata.assign(avg_accuracy = average_accuracies.groupby(['installation_id']).mean())
        #tdata = tdata.assign(avg_accuracy_group = all_accuracy_groups.groupby(['installation_id']).mean())

        for title in list(dict.fromkeys(d.data_frame['title'].values)):
            if title in list(dict.fromkeys(assessment['title'].values)):
                assessment[f'{title} acc'] = (assessment['title'] == title)
                assessment[f'{title} acc_group'] = assessment[f'{title} acc']
                tdata[f'{title} acc'] = assessment.groupby(['installation_id'], sort = False).sum()[f'{title} acc']
                tdata[f'{title} acc_group'] = tdata[f'{title} acc']
                tdata.fillna(0, inplace = True)
                tdata[f'{title} acc'] = average_accuracies.reset_index()[average_accuracies.reset_index()['title'] == title].groupby(['installation_id'], sort = False).sum()['success'].divide(tdata[f'{title} acc']).fillna(0)
                tdata[f'{title} acc_group'] = all_accuracy_groups.reset_index()[all_accuracy_groups.reset_index()['title'] == title].groupby(['installation_id'], sort = False).sum()['success'].divide(tdata[f'{title} acc_group']).fillna(0)
                #tdata[f'{title} attempts'] = total_attempts.reset_index()[average_accuracies.reset_index()['title'] == title].groupby(['installation_id'], sort=False).mean()['success'].divide(tdata[f'{title} acc']).fillna(0)
            tdata[f'{title} avg time'] = d.data_frame[d.data_frame['title'] == title].groupby(['installation_id', 'game_session'], sort=False).apply(lambda g: g['game_time'].max()).reset_index().groupby(['installation_id'], sort=False).mean()
            tdata[f'{title} std time'] = d.data_frame[d.data_frame['title'] == title].groupby(['installation_id', 'game_session'], sort=False).apply(lambda g: g['game_time'].max()).reset_index().groupby(['installation_id'], sort=False).std()

        tdata['correct_attempts'] = correct_attempts.reset_index().groupby(['installation_id'], sort=False).sum()['success']
        tdata['incorrect_attempts'] = correct_attempts.reset_index().groupby(['installation_id'], sort=False).count()['success']-tdata['correct_attempts']

        tdata = tdata.join(pd.get_dummies(all_accuracy_groups.reset_index(drop = True)).assign(installation_id = all_accuracy_groups.reset_index()['installation_id']).groupby(['installation_id'], sort = False).sum())
        tdata = tdata.join((pd.get_dummies(d.data_frame['type']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))
        tdata = tdata.join((pd.get_dummies(d.data_frame['world']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))
        tdata = tdata.join((pd.get_dummies(d.data_frame['event_id']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))
        tdata = tdata.join((pd.get_dummies(d.data_frame['event_code'].astype(str)+d.data_frame['title']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))
        tdata = tdata.join((pd.get_dummies(d.data_frame['title']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))
        tdata = tdata.join((pd.get_dummies(d.data_frame['event_code']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))

        if not test:
            tdata.index = tdata.index + str(count)
            count += 1

            if first:
                data = tdata.copy()
                first = False
            else:
                data = data.append(tdata)
        
        if test:
            break

    if not test:
        tdata = data.fillna(0)
        del(data)

    tdata = tdata.replace([np.inf, -np.inf], np.nan)
    tdata.fillna(0, inplace=True)

    return tdata

def drop_rare_variables(min_occurence):
    events = (pd.get_dummies(d.data_frame['event_id']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int)
    events = events.join((pd.get_dummies(d.data_frame['event_code'].astype(str)+d.data_frame['title']).assign(installation_id = d.data_frame['installation_id']).groupby(['installation_id'], sort = False).sum()).astype(int))
    events = (events > 0).astype(int).sum()
    drop = []

    for event in events.index.values:
        if events.loc[event] < 0:
            drop.append(event)

    tdata.drop(labels = drop , axis = 1, inplace = True)

def feature_selection(df):
    cols = list(df.columns.values)
    cols.remove('accuracy_group')
    new = cols.copy()
    reduced = cols.copy

    removed = []
    dropped = 0
    count = 2
    for col in tqdm(cols):
        output = k_fold_validation(df[new], df['accuracy_group'], 10)
        new = reduced.copy()
        new.remove(col)
        if col == cols[0]:
            score = output['kappa']
            continue
        if output['kappa']>=score:
            reduced.remove(col)
            removed.append(col)
            score = output['kappa']
        print(f'{count} of {len(cols)}')
        count+=1

    return reduced

def get_features(test, train, adjust = False):
    features = list(test.columns.values)
    out = features.copy()

    for feat in features:
        if test[feat].sum() == 0:
            out.remove(feat)
            continue

        testmean = test[feat].mean()
        trainmean = train[feat].mean()
        ratio = trainmean/testmean

        if ratio > 10 or ratio < 0.1:
            out.remove(feat)
        elif ('final assessment' not in str(feat)) and adjust:
            test[feat]*=ratio

        if test[feat].sum() == 0:
            out.remove(feat)

    return out, test

def drop_high_corr(data):
    feats = list(data.columns.values)
    feats.remove('accuracy_group')
    drop = []
    counter = 0
    for feat1 in feats:
        for feat2 in feats:
            if feat1!=feat2 and feat1 not in drop and feat2 not in drop:
                corr = np.corrcoef(data[feat1], data[feat2])[0][1]
                if corr > 0.995:
                    counter += 1
                    drop.append(feat2)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat1, feat2, corr))
    return drop

test = get_data(test = True, path = '/kaggle/input/data-science-bowl-2019/test.csv')
train = pd.read_csv('/kaggle/input/mysolution/tdata_reduced.csv')
train.set_index('installation_id', inplace = True)
#train = train[(train.index.str.len()==9) & (train.index.str.strip().str[-1].astype(int) == 1)]

exclude = ['Air Show avg time','All Star Sorting avg time', 'Fireworks (Activity) avg time', 'assessment_Bird Measurer (Assessment) final assessment', 'avg_accuracy', 'assessment_Bird Measurer (Assessment) final assessment', 'avg_accuracy_group', '12 Monkeys avg time', '12 Monkeys std time', 'Costume Box avg time', 'Costume Box std time', 'Magma Peak - Level 1 avg time', 'Magma Peak - Level 1 std time', 'Ordering Spheres avg time', 'Ordering Spheres std time', "Pirate's Tale avg time", "Pirate's Tale std time", 'Rulers avg time', 'Rulers std time', 'Slop Problem avg time', 'Slop Problem std time', 'Treasure Map avg time', 'Treasure Map std time', 'Tree Top City - Level 1 avg time', 'Tree Top City - Level 1 std time', 'Tree Top City - Level 2 avg time', 'Tree Top City - Level 2 std time', 'Tree Top City - Level 3 avg time', 'Tree Top City - Level 3 std time', 'Welcome to Lost Lagoon! avg time', 'Welcome to Lost Lagoon! std time', 'Air Show avg time', 'assessment_Bird Measurer (Assessment) final assessment', 'avg_accuracy', 'avg_accuracy_group', 'event_code_count', 'Cart Balancer (Assessment) attempts', '01ca3a3c', '0ce40006', '119b5b02', '1325467d', '1b54d27f', '31973d56', '36fa3ebe', '6077cc36', '611485c5', '7040c096', 'ab4ec3a4', 'Mushroom Sorter (Assessment) attempts', 'All Star Sorting avg time', 'total_time', 'Bird Measurer (Assessment) acc', 'Bird Measurer (Assessment) avg time', '16667cc5', '222660ff', '25fa8af4', '3edf6747', '56cd3b43', '6043a2b4', '6aeafed4', '4110']
ds = [c for c in list(train.columns.values) if c not in exclude]
train = train[ds]
headers = list(train.columns.values)
headers.remove('accuracy_group')
features = headers.copy()

for header in headers:
    if header not in test.columns.values and header != 'installation_id':
        features.remove(header)
        
train_feats = features.copy()
train_feats.append('accuracy_group')
test = test[features]
train = train[train_feats]

feats, test = get_features(test, train, adjust = False)

#nn_m = nn_model()
#xgb_m = xgb_model()
lgb_m = lgb_model()

for model in [lgb_m]:
    model.convert = False
    model.fit(train[feats].values, train['accuracy_group'].values)

ensemble_model = ensemble()

#ensemble_model.add_model(xgb_m, weight = 0.10)
#ensemble_model.add_model(nn_m, weight = 0.00)
ensemble_model.add_model(lgb_m, weight = 1.0)

predictions = ensemble_model.predict(test[feats].values)

dist = Counter(train[(train.index.str.len()==9) & (train.index.str.strip().str[-1].astype(int) == 1)]['accuracy_group'])
for k in dist:
    dist[k] /= len(train['accuracy_group'])
acum = 0
bound = {}
for i in range(3):
    acum += dist[i]
    bound[i] = np.percentile(predictions, acum * 100)

def classify(x):
    if x <= bound[0]:
        return 0
    elif x <= bound[1]:
        return 1
    elif x <= bound[2]:
        return 2
    else:
        return 3

predictions = np.array(list(map(classify, predictions)))

ids = list(dict.fromkeys(test.index.values))

submission = pd.DataFrame({'installation_id': ids, 'accuracy_group': predictions})
submission.to_csv('submission.csv', index = False)