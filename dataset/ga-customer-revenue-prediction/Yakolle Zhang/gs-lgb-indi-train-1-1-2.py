import sys

import lightgbm as lgb
from scipy.sparse import hstack, vstack
from sklearn.model_selection import GroupShuffleSplit

from cetune.cv_util import *

id_col = 'fullVisitorId'
kfold_k = 5
 
oof_seed = 1
pred_type_id = 1
indi_type_id = 2
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]
user_mode = indi_type_id > 2
pred_max_val = np.log1p(7.8e10) if user_mode else np.log1p(2.4e10)
need_embeded_target = pred_type_id in [4, 6]


def auc(y, p):
    return metrics.roc_auc_score(y, p)


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def refine_embeded_target(df):
    cols = ['min_target', 'max_target', 'mean_target']
    if need_embeded_target:
        for col in cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
    else:
        df = df.drop(cols, axis=1, errors='ignore')
    return df


def get_oof_data(data_dir):
    with timer('get train src data'):
        train_df = pd.read_pickle(os.path.join(data_dir, 'train_df'), compression='gzip')
        train_df_u = pd.read_pickle(os.path.join(data_dir, 'train_df_u'), compression='gzip').reset_index()
        train_df = refine_embeded_target(train_df)
        gc.collect()
        train_df_u = refine_embeded_target(train_df_u)
        gc.collect()

        tr_x_text = joblib.load(os.path.join(data_dir, 'tr_x_text_u')) \
            if user_mode else joblib.load(os.path.join(data_dir, 'tr_x_text'))
        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        print(train_df.shape, train_df_u.shape, tr_x_text.shape)

    with timer('oof split'):
        iind, oind = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(
            train_df, groups=train_df[id_col]))
        if user_mode:
            ids = train_df.take(iind)[id_col]
            iind = train_df_u.loc[train_df_u[id_col].isin(ids)].index.values
            ids = train_df.take(oind)[id_col]
            oind = train_df_u.loc[train_df_u[id_col].isin(ids)].index.values
            train_df = train_df_u
        in_df = train_df.take(iind)
        in_x_text = tr_x_text[iind]
        print(in_df.shape, in_x_text.shape)

        out_df = train_df.take(oind)
        out_x_text = tr_x_text[oind]
        print(out_df.shape, out_x_text.shape)
        del train_df, tr_x_text, train_df_u, iind, oind
        gc.collect()

    with timer('combine features'):
        in_selector = in_df.loc[:, ['fullVisitorId']].reset_index(drop=True)
        iy = np.log1p(in_df.target.values) if 1 == indi_type_id else (in_df.target.values > 0).astype(np.int8)
        ix = in_df.drop(['fullVisitorId', 'target'], axis=1).values
        ix = combine_features([ix, in_x_text]) if 'plain' != text_type else ix
        del in_df, in_x_text
        gc.collect()

        out_selector = out_df.loc[:, ['fullVisitorId']].reset_index(drop=True)
        oy = np.log1p(out_df.target.values) if 1 == indi_type_id else (out_df.target.values > 0).astype(np.int8)
        ox = out_df.drop(['fullVisitorId', 'target'], axis=1).values
        ox = combine_features([ox, out_x_text]) if 'plain' != text_type else ox
        del out_df, out_x_text
        gc.collect()

        print(ix.shape, ox.shape)

    return ix, iy, in_selector, ox, oy, out_selector


def get_test_data(data_dir):
    with timer('get test src data'):
        test_df = pd.read_pickle(os.path.join(data_dir, 'test_df_u'), compression='gzip').reset_index() \
            if user_mode else pd.read_pickle(os.path.join(data_dir, 'test_df'), compression='gzip')
        test_df = refine_embeded_target(test_df)
        gc.collect()

        ts_x_text = joblib.load(os.path.join(data_dir, 'ts_x_text_u')) \
            if user_mode else joblib.load(os.path.join(data_dir, 'ts_x_text'))
        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        print(test_df.shape, ts_x_text.shape)

    with timer('combine features'):
        ts_selector = test_df.loc[:, ['fullVisitorId']].reset_index(drop=True)
        ts_x = test_df.drop('fullVisitorId', axis=1).values
        ts_x = combine_features([ts_x, ts_x_text]) if 'plain' != text_type else ts_x
        print(ts_x.shape)
        del test_df, ts_x_text
        gc.collect()

    return ts_x, ts_selector


def learn(data, model_dir='.'):
    indi_id = f'indi_{oof_seed}_{pred_type_id}_{indi_type_id}'
    ix, iy, in_selector, ox, oy, out_selector, ts_x, ts_selector = data

    with timer('train'):
        seed = oof_seed * 1000000 + pred_type_id * 10000 + indi_type_id
        fold_inds = kfold((ix, iy), kfold_k, True, seed) if user_mode else group_kfold((ix, iy, in_selector, id_col),
                                                                                       kfold_k, True, seed)

        if 1 == indi_type_id:
            params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4,
                      'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.8,
                      'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
        else:
            params = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'nthread': 4,
                      'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.8,
                      'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0, 'is_unbalance': True}

        ip = np.zeros(iy.shape)
        ops = []
        ps = []
        t_scores = []
        v_scores = []
        o_scores = []
        measure_func = rmse if 1 == indi_type_id else auc
        measure_name = measure_func.__name__
        for tind, vind in fold_inds:
            tx, ty, vx, vy = ix[tind], iy[tind], ix[vind], iy[vind]
            print(tx.shape, vx.shape)
            model = lgb.train(params, lgb.Dataset(tx, label=ty), 100000, [lgb.Dataset(vx, label=vy)],
                              early_stopping_rounds=200, verbose_eval=200)
            model_file_name = write_model(model, model_dir)
            print(f"{model_file_name}'s best iteration: {model.best_iteration}")

            tp, vp, op, p, t_score, v_score, o_score = validation((tx, ty, vx, vy, ox, oy, ts_x), model, measure_func)
            ip[vind] = vp
            ops.append(op)
            ps.append(p)
            t_scores.append(t_score)
            v_scores.append(v_score)
            o_scores.append(o_score)
            del tx, ty, vx, vy
            gc.collect()
        print(f't_{measure_name}s: mean={np.mean(t_scores)}, std={np.std(t_scores)}, {t_scores}')
        print(f'v_{measure_name}s: mean={np.mean(v_scores)}, std={np.std(v_scores)}, {v_scores}')
        print(f'o_{measure_name}s: mean={np.mean(o_scores)}, std={np.std(o_scores)}, {o_scores}')

        in_selector['pred'] = ip
        in_selector.to_pickle(f'{indi_id}_ip', compression='gzip')
        out_selector['pred'] = np.mean(ops, axis=0)
        out_selector.to_pickle(f'{indi_id}_op', compression='gzip')
        ts_selector['pred'] = np.mean(ps, axis=0)
        ts_selector.to_pickle(f'{indi_id}_p', compression='gzip')


def validation(data, model, measure_func):
    tx, ty, vx, vy, ox, oy, ts_x = data
    measure_name = measure_func.__name__

    with timer('validation'):
        tp = model.predict(tx)
        tp = np.clip(tp, 0, pred_max_val) if 1 == indi_type_id else tp
        t_score = measure_func(ty, tp)
        vp = model.predict(vx)
        vp = np.clip(vp, 0, pred_max_val) if 1 == indi_type_id else vp
        v_score = measure_func(vy, vp)
        op = model.predict(ox)
        op = np.clip(op, 0, pred_max_val) if 1 == indi_type_id else op
        o_score = measure_func(oy, op)
        p = model.predict(ts_x)
        p = np.clip(p, 0, pred_max_val) if 1 == indi_type_id else p
        print(f't_{measure_name}: {t_score}, v_{measure_name}: {v_score}, o_{measure_name}: {o_score}')

    return tp, vp, op, p, t_score, v_score, o_score


def run():
    sys.stderr = sys.stdout = open(os.path.join('lgb_indi_train_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-src-data'
    print(os.listdir(data_dir))

    ix, iy, in_selector, ox, oy, out_selector = get_oof_data(data_dir)
    ts_x, ts_selector = get_test_data(data_dir)
    data = ix, iy, in_selector, ox, oy, out_selector, ts_x, ts_selector
    learn(data)


if __name__ == '__main__':
    run()
