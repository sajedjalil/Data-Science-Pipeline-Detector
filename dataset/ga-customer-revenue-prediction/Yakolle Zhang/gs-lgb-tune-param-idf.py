import sys
import warnings

import lightgbm as lgb
from scipy.sparse import hstack, vstack

from cetune.tune_util import *

warnings.filterwarnings('ignore')
start_time = int(time.time())
end_time = start_time + 9 * 3600 - 20 * 60

text_type = 'idf'
pred_max_val = np.log1p(5.1e10)
num_round = 600


def get_values(x):
    return x.values if hasattr(x, 'values') else x


class LgbTrainer:
    def __init__(self, params):
        self.params = params
        self.model = None

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, x, y):
        self.model = lgb.train(self.params, lgb.Dataset(get_values(x), label=get_values(y)),
                               num_boost_round=self.params['num_boost_round'])
        return self

    def predict(self, x):
        return self.model.predict(get_values(x))


def rmse(y, p):
    p = np.clip(p, 0, pred_max_val)
    return metrics.mean_squared_error(y, p) ** 0.5


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def get_data(data_dir):
    with timer('get train src data'):
        train_df = pd.read_pickle(os.path.join(data_dir, 'train_df'), compression='gzip')
        tr_x_text = joblib.load(os.path.join(data_dir, 'tr_x_text'))
        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        print(f'train_df: {train_df.shape}, tr_x_text: {tr_x_text.shape}')

    with timer('log transform'):
        embed_cols = [col for col in train_df.columns if '_price_w_' in col or '_price_u_' in col
                      or 'Revenue_w_' in col or 'Revenue_u_' in col or 'target_u_' in col]
        print(f'embed_cols({len(embed_cols)}): {embed_cols}')
        for col in embed_cols:
            train_df[col] = np.log1p(train_df[col])
        gc.collect()

    with timer('combine features'):
        y = np.log1p(train_df.target.values)
        x = train_df.drop(['fullVisitorId', 'target'], axis=1).values
        x = combine_features([x, tr_x_text]) if 'plain' != text_type else x
        del train_df, tr_x_text
        gc.collect()
        print(f'x: {x.shape}, y(>0): {np.sum(y)}')

    return x, y


def tune_params(data_dir, tune_dir='.'):
    x, y = get_data(data_dir)

    params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4, 'num_boost_round': num_round,
              'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.8,
              'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
    model = LgbTrainer(params=params)

    init_param = [('learning_rate', 0.01), ('num_leaves', 31), ('max_depth', 0), ('min_data', 20),
                  ('bagging_fraction', 0.8), ('feature_fraction', 0.8), ('lambda_l1', 0), ('lambda_l2', 0)]
    param_dic = {'learning_rate': [.008, .02, .04, .08],
                 'num_leaves': [20, 40, 80],
                 'max_depth': [0, 8, 10, 12, 14],
                 'min_data': [20, 40, 80, 160],
                 'bagging_fraction': ['grid', .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
                 'feature_fraction': ['grid', .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
                 'lambda_l1': [0.0, .01, .02, .04, .08, .2, .4, .8],
                 'lambda_l2': [0.0, .01, .02, .04, .08, .2, .4, .8]}

    tune(model, (x, y), init_param, param_dic, measure_func=rmse, detail=True, random_state=6000, data_dir=tune_dir,
         kc=(3, 1), max_optimization=False, score_min_gain=1e-3, kfold_func=kfold,
         task_id=f'lgb_{text_type}_{num_round}', non_ordinal_factors=['max_depth'], end_time=end_time)


def run():
    sys.stderr = sys.stdout = open(os.path.join(f'lgb_{text_type}_tune_params_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-src-data'
    print(os.listdir(data_dir))
   
    task_id = f'lgb_{text_type}_{num_round}'
    cache_root_dir = f'../input/gs-lgb-tune-param-{text_type}-reboot'
    print(os.listdir(cache_root_dir))
    shutil.copytree(os.path.join(cache_root_dir, 'cache', task_id), os.path.join('.', 'cache', task_id))

    tune_params(data_dir)


if __name__ == '__main__':
    run()
