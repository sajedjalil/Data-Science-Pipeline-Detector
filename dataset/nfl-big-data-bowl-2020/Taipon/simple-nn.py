print('loading libs...')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import random
import tqdm
import keras
import datetime
from keras import backend as K
from keras import callbacks
from scipy import stats
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.competitions import nflrush
print('done')

# loading env
env = nflrush.make_env()

# some constants to be used
DF_NAME= []
PATH = '../input/nfl-big-data-bowl-2020/'
SEED = 1229
NFOLDS = 5
NREPEATS = 5

# some functions to be used

# func for seeding
def SeedEverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    return

# func for loading data
def DataLoading(path, df_name):
    files = os.listdir(f'{path}')
    for i in range(len(files)):
        s0 = files[i]
        s1 = files[i][:-4]
        s2 = files[i][-4:]
        if s2 =='.csv':
            print('loading:'+ s1 + '...')
            globals()[s1] = pd.read_csv(f'{path}'+ s0,  dtype={'WindSpeed': 'object'})
            df_name.append(s1)
        elif s2 == '.pkl':
            print('loading:'+ s1 + '...')
            globals()[s1] = pd.read_pickle(f'{path}'+ s0,  dtype={'WindSpeed': 'object'})
            df_name.append(s1)
        else:
            pass
    print('successfully loading: ')
    print(df_name)
    print('done')
    return df_name

# func for data analysis
def DataStatistics(df):   
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values 
    summary['Missing_percentage'] = round((summary['Missing']/df.shape[0])*100, 1)
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values
       
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    summary.set_index('Name',inplace=True)
    summary = summary.T
    return summary

# func for showing data
def DataShowing(df_name, start=0, end=49, seeall = True):
    if seeall:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)   
    df_name.sort(reverse=True)
    for i in range(len(df_name)):
        s = df_name[i]
        df = globals()[s]
        print('data shape of ' + s + ':' + f'{df.shape}')
        df = df.iloc[:,start:end]
        if df.empty:
            pass
        else:
            print('looking over the statistics of all features of ' + s + '...')
            display(DataStatistics(df))
            print('looking over the statistics of the num_type_features of ' + s + '...')
            display(df.describe())
        return

# func for processing training data 
def DataProcessing(train):
    train['PossessionTeam'] = train['PossessionTeam'].map(map_abbr)
    train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(map_abbr)
    train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(map_abbr)
    train['HomePossesion'] = train['PossessionTeam'] == train['HomeTeamAbbr']
    
    train = pd.concat([train.drop(['OffenseFormation'], axis=1), 
                      pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)
    globals()['dummy_col'] = train.columns
    train['GameClock'] = train['GameClock'].apply(strtoseconds)
    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60*60*24*365.25
    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)
    train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
    train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')
    train['GameWeather'] = train['GameWeather'].str.lower()
    indoor = "indoor"
    train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    train['GameWeather'] = train['GameWeather'].apply(map_weather)
    train['IsRusher'] = train['NflId'] == train['NflIdRusher']
    train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection','NflId', 'NflIdRusher'],
               axis=1, inplace=True)
    train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
    train.drop(['GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1, inplace=True)
    cat_features = []
    for col in train.columns:
        if train[col].dtype =='object':
            cat_features.append(col)
    train.drop(cat_features, axis=1,inplace=True)
    train.fillna(-999, inplace=True)
    globals()['players_col'] = []
    for col in train.columns:
        if train[col][:22].std()!=0:
            players_col.append(col)    
    X_train = np.array(train[players_col]).reshape(-1, 11*22)
    play_col = train.drop(players_col+['Yards'], axis=1).columns
    X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))
    for i, col in enumerate(play_col):
        X_play_col[:, i] = train[col][::22]
    X_train = np.concatenate([X_train, X_play_col], axis=1)
    y_train = np.zeros(shape=(X_train.shape[0], 199))
    for i,yard in enumerate(train['Yards'][::22]):
        y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))
    return X_train, y_train
    
    

# func for convert str to seconds   
def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

# func for convert str to float 
def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1
    
# func for map weather    
def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

# func for metric
def crps(y_true, y_pred):
    ans = 0
    ground_t = y_true.argmax(1)
    for i, t in enumerate(ground_t):
        for n in range(-99, 100):
            h = n>=(t-99)
            
            ans+=(y_pred[i][n+99]-h)**2
            
    return ans/(199*len(y_true))

# RAdam class

__all__ = ['RAdam']
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(min_lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)
            decay_rate = (self.min_lr - lr) / decay_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))
            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),           
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# func for building the model
def ModelBuild(X_train, y_train):
    score = 0
    i = 0
    folds = RepeatedKFold(n_splits=NFOLDS,  n_repeats=NREPEATS, random_state = SEED)
    splits = folds.split(X_train,y_train)
    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_tr, X_val = X_train[train_index], X_train[valid_index]
        y_tr, y_val = y_train[train_index], y_train[valid_index]
        
        globals()['model' + str(i)] = keras.models.Sequential([
            keras.layers.Dense(units=512, input_shape=[X_tr.shape[1]], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(units=256, activation='relu'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=256, activation='relu'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(units=199, activation='sigmoid')
            ])
        globals()['model' + str(i)] .compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), loss='mse')
        es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15,verbose=1, mode='auto', restore_best_weights=True)
        rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10, min_lr=1e-6, mode='auto', verbose=1)
        globals()['model' + str(i)].fit(X_tr,[y_tr], validation_data=(X_val,[y_val]), callbacks=[es, rlr],epochs=100, batch_size=128, verbose=0)
        cv_predict= globals()['model' + str(i)].predict(X_val)
        crps_res = crps(y_val,cv_predict)
        print(f'The CV score is: {crps_res}')
        score +=crps_res      
        i += 1
    score = score/(NFOLDS*NREPEATS)
    print(f'The averge CV score is: {score}')
    return
    

# func for make prediction
def make_pred(df, sample, env, model):
    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)
    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)
    missing_cols = set( dummy_col ) - set( test.columns )-set('Yards')
    for c in missing_cols:
        df[c] = 0
    df = df[dummy_col]
    df.drop(['Yards'], axis=1, inplace=True)
    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    df['GameClock'] = df['GameClock'].apply(strtoseconds)
    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60*60*24*365.25
    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)
    df['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')
    indoor = "indoor"
    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(map_weather)
    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    
    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1)
    cat_features = []
    for col in df.columns:
        if df[col].dtype =='object':
            cat_features.append(col)

    df = df.drop(cat_features, axis=1)
    df.fillna(-999, inplace=True)
    X = np.array(df[players_col]).reshape(-1, 11*22)
    play_col = df.drop(players_col, axis=1).columns
    X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))
    for i, col in enumerate(play_col):
        X_play_col[:, i] = df[col][::22]
    X = np.concatenate([X, X_play_col], axis=1)
    y_pred = np.zeros(199).reshape(1,199)
    for i in range(NFOLDS*NREPEATS):
        globals()['y_pred' + str(i)] = globals()['model' + str(i)].predict(X)
        y_pred += globals()['y_pred' + str(i)]
    y_pred = y_pred/(NFOLDS*NREPEATS)
    for pred in y_pred:
        prev = 0
        for i in range(len(pred)):
            if pred[i]<prev:
                pred[i]=prev
            prev=pred[i]
    y_pred = pd.DataFrame(data=y_pred,columns=sample.columns)
    #y_pred.iloc[:,:85] = 0
    #y_pred.iloc[:,130:] = 1
    env.predict(y_pred)
    return y_pred

# seeding 
SeedEverything(SEED)

# loading data
df_name = DataLoading(PATH, DF_NAME)

# some variables to be used
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train['PossessionTeam'].unique():
    map_abbr[abb] = abb
off_form = train['OffenseFormation'].unique()

# processing traing data
X_train, y_train = DataProcessing(train)

# building the NN model
model = ModelBuild(X_train, y_train)

# making predictions
for test, sample in tqdm.tqdm(env.iter_test()):
    make_pred(test, sample, env, model)
    
    # submitting
env.write_submission_file()