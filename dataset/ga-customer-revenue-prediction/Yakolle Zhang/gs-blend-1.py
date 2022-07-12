import sys
import warnings

import lightgbm as lgb

from cetune.tune_util import *

warnings.filterwarnings('ignore')

root_dir = '../input'
pred_max_val = np.log1p(5.1e10)
oof_seed = 1
pred_type_ids = [1, 4, 8]
text_types = ['idf', 'bin', 'plain']
need_tune = True


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


class AvgEnsembler:
    def __init__(self, model_ids):
        self.model_ids = model_ids
        self.weights = np.ones(len(model_ids))

    def set_params(self, **params):
        for k, v in params.items():
            self.weights[self.model_ids.index(k)] = v

    def fit(self, px, py):
        return self

    def predict(self, px):
        return np.dot(px, self.weights / np.sum(self.weights))


def rmse(y, p):
    p = np.clip(p, 0, pred_max_val)
    return metrics.mean_squared_error(y, p) ** 0.5


def submit(pred, sub_data_dir, submit_id):
    with timer('submit'):
        submission = pd.read_csv(os.path.join(sub_data_dir, 'sample_submission_v2.csv'), dtype={'fullVisitorId': 'str'})
        submission['PredictedLogRevenue'] = pred
        submission.to_csv(f'{submit_id}.csv', index=False, float_format='%.5f')


def get_model_type(pred_type_id):
    model_type_id = (pred_type_id - 1) // len(text_types) + 1
    if 1 == model_type_id:
        return 'lgb'
    else:
        return 'nn'


def get_pred(pred_type_id):
    pred_dir = os.path.join(root_dir, f'gs-{get_model_type(pred_type_id)}-uv-ensemble-{oof_seed}-{pred_type_id}')
    model_id = f'{get_model_type(pred_type_id)}_uv_ensemble_{oof_seed}_{pred_type_id}'
    oy = joblib.load(os.path.join(pred_dir, f'{model_id}_oy'))
    op = joblib.load(os.path.join(pred_dir, f'{model_id}_op'))
    p = joblib.load(os.path.join(pred_dir, f'{model_id}_p'))
    return oy, op, p, model_id


def get_preds():
    oy = None
    ops = []
    ps = []
    model_ids = []
    for i in pred_type_ids:
        oy, op, p, model_id = get_pred(i)
        ops.append(op)
        ps.append(p)
        model_ids.append(model_id)

    return np.array(ops).T, oy, np.array(ps).T, model_ids


def avg_tune(px, py, model_ids):
    with timer('tune'):
        model = AvgEnsembler(model_ids)

        init_param = [(model_id, 1.0) for model_id in model_ids]
        param_dic = dict([(model_id, arange(0, 1.1, 0.1)) for model_id in model_ids])
        # init_param = [(model_id, 1.0) for model_id in model_ids]
        # param_dic = dict([(model_id, ['grid'] + arange(0, 1.01, 0.01)) for model_id in model_ids])

        w, _ = tune(model, (px, py), init_param, param_dic, measure_func=rmse, detail=True, random_state=oof_seed + 1,
                    factor_cache={}, kc=(3, 1), max_optimization=False, score_min_gain=1e-3)
        print(w)

    with timer('validation'):
        model.set_params(**dict(w))
        pp = model.predict(px)
        print(f'o_rmse: {rmse(py, pp)}')

    return w


def avg(submit_id):
    with timer('load preds'):
        opx, opy, px, model_ids = get_preds()
        print(f'opx: {opx.shape}, px: {px.shape}, model_ids: {len(model_ids)}')

    w = avg_tune(opx, opy, model_ids)

    with timer('predict'):
        ensembler = AvgEnsembler(model_ids)
        ensembler.set_params(**dict(w))
        pp = ensembler.predict(px)
        joblib.dump(pp, f'{submit_id}_p', compress=3)
        print(f'pp: {pp.shape}')

    return pp


def blend_tune(px, py):
    params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4,
              'num_boost_round': 1000, 'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20,
              'bagging_fraction': 0.8, 'feature_fraction': 1.0, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
    model = LgbTrainer(params=params)

    init_param = [('learning_rate', 0.01), ('num_leaves', 31), ('max_depth', 0), ('min_data', 20),
                  ('bagging_fraction', 0.9), ('lambda_l1', 0), ('lambda_l2', 0)]
    param_dic = {'learning_rate': [.04, .08, .2, .4],
                 'num_leaves': [20, 40, 80],
                 'max_depth': [4, 8, 10, 12, 14],
                 'min_data': [20, 40, 80, 160],
                 'bagging_fraction': ['grid', .4, .5, .6, .7, .8, .9, 1.0],
                 'lambda_l1': [0.0, .04, .08, .2, .4, .8],
                 'lambda_l2': [0.0, .04, .08, .2, .4, .8]}

    params, _ = tune(model, (px, py), init_param, param_dic, measure_func=rmse, detail=True, random_state=oof_seed + 2,
                     factor_cache={}, kc=(3, 1), max_optimization=False, score_min_gain=1e-3)
    print(params)

    with timer('validation'):
        model.set_params(**dict(params))
        model.fit(px, py)
        pp = model.predict(px)
        print(f'o_rmse: {rmse(py, pp)}')

    return params


def blend(submit_id):
    with timer('load preds'):
        opx, opy, px, model_ids = get_preds()
        print(f'opx: {opx.shape}, px: {px.shape}, model_ids: {len(model_ids)}')

    params = [('learning_rate', 0.0095787047), ('num_leaves', 2), ('max_depth', 0), ('min_data', 130),
              ('bagging_fraction', 0.9), ('lambda_l1', 0.08), ('lambda_l2', 0.08)]
    if need_tune:
        params = blend_tune(opx, opy)

    with timer('blender train'):
        run_params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4,
                      'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.9,
                      'feature_fraction': 1.0, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
        run_params.update(dict(params))

        blender = lgb.train(run_params, lgb.Dataset(opx, label=opy), 1000)
        write_model(blender, '.', model_id=submit_id)

    with timer('validation'):
        opp = blender.predict(opx)
        print(f'o_rmse: {rmse(opy, opp)}')

        print(model_ids)
        print(f'importance(split): {blender.feature_importance(importance_type="split")}')
        print(f'importance(gain): {blender.feature_importance(importance_type="gain")}')

    with timer('predict'):
        pp = np.clip(blender.predict(px), 0, pred_max_val)
        joblib.dump(pp, f'{submit_id}_p', compress=3)
        print(f'pp: {pp.shape}')

    return pp


def run():
    sys.stderr = sys.stdout = open(os.path.join('blend_log'), 'w')
    sub_data_dir = '../input/ga-customer-revenue-prediction'

    submit_id = f'avg_{oof_seed}'
    avg_p = avg(submit_id)
    submit(avg_p, sub_data_dir, submit_id)

    submit_id = f'blend_{oof_seed}'
    blend_p = blend(submit_id)
    submit(blend_p, sub_data_dir, submit_id)

    submit_id = f'average_{oof_seed}'
    average_p = 0.5 * avg_p + 0.5 * blend_p
    joblib.dump(average_p, f'{submit_id}_p', compress=3)
    submit(average_p, sub_data_dir, submit_id)


if __name__ == '__main__':
    run()
