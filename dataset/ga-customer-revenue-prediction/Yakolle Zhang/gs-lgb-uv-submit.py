import gc

import lightgbm as lgb
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit

from cetune.util import *

id_col = 'fullVisitorId'

oof_seed = 1
pred_type_id = 1
text_types = ['idf', 'bin', 'plain']
text_type = text_types[(pred_type_id - 1) % len(text_types)]
pred_max_val = np.log1p(7.8e10)
need_prune_combo_cols = False


def rmse(y, p):
    return metrics.mean_squared_error(y, p) ** 0.5


def combine_features(features, batch_num=5):
    cols = []
    batch_size = features[0].shape[0] // batch_num + 1
    for i in range(batch_num):
        fts = [ft[i * batch_size: (i + 1) * batch_size] for ft in features]
        cols.append(hstack(fts, dtype=np.float32).tocsr())
    return vstack(cols)


def prune_combo_cols(df):
    ratio_cols = [col for col in df.columns if '_ratio' in col]
    sum_cols = [col.replace('ratio', 'sum') for col in ratio_cols]
    mean_cols = [col.replace('ratio', 'mean') for col in ratio_cols]
    return df.drop(sum_cols + mean_cols, axis=1)


def submit(pred, ids, sub_data_dir, submit_id):
    with timer('submit'):
        pred = pd.Series(pred, index=pd.Index(ids, name='fullVisitorId'), name='pred')

        submission = pd.read_csv(os.path.join(sub_data_dir, 'sample_submission.csv'), dtype={'fullVisitorId': 'str'})
        submission = submission.join(pred, on='fullVisitorId')
        submission['PredictedLogRevenue'] = submission['pred']
        print(f'PredictedLogRevenue.isnull: {np.sum(submission["PredictedLogRevenue"].isnull())}')

        submission = submission.drop('pred', axis=1)
        submission.to_csv(f'{submit_id}.csv', index=False, float_format='%.5f')


def get_indicators(root_dir='../input'):
    def agg_num_by_user(df):
        df['pred'] = np.expm1(df['pred'])
        gdf = df.groupby(id_col)['pred'].agg([np.sum, np.mean, np.std]).rename(
            columns={'sum': 'zr_sum', 'mean': 'zr_mean', 'std': 'zr_std'})
        gdf['zr_log_sum'] = np.log1p(gdf['zr_sum'])
        gdf['zr_mean_low'] = gdf['zr_mean'] - gdf['zr_std']
        gdf['zr_mean_high'] = gdf['zr_mean'] + gdf['zr_std']
        return gdf

    def agg_indi_by_user(df):
        gdf = df.groupby(id_col)['pred'].agg([np.mean, np.std]).rename(columns={'mean': 'zc_mean', 'std': 'zc_std'})
        gdf['zc_mean_low'] = gdf['zc_mean'] - gdf['zc_std']
        gdf['zc_mean_high'] = gdf['zc_mean'] + gdf['zc_std']
        return gdf

    with timer('load regression indi'):
        indi_type_id = 1
        indi_id = f'indi_{oof_seed}_{pred_type_id}_{indi_type_id}'
        indi_dir = f'gs-lgb-indi-train-{oof_seed}-{pred_type_id}-{indi_type_id}'

        in_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'), compression='gzip')
        in_indi_df = agg_num_by_user(in_selector)
        out_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_op'), compression='gzip')
        out_indi_df = agg_num_by_user(out_selector)
        ts_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_p'), compression='gzip')
        ts_indi_df = agg_num_by_user(ts_selector)
        print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
        del in_selector, out_selector, ts_selector
        gc.collect()

    with timer('load classification(session) indi'):
        indi_type_id = 2
        indi_id = f'indi_{oof_seed}_{pred_type_id}_{indi_type_id}'
        indi_dir = f'gs-lgb-indi-train-{oof_seed}-{pred_type_id}-{indi_type_id}'

        in_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'), compression='gzip')
        in_indi_df = in_indi_df.join(agg_indi_by_user(in_selector), how='inner')
        out_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_op'), compression='gzip')
        out_indi_df = out_indi_df.join(agg_indi_by_user(out_selector), how='inner')
        ts_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_p'), compression='gzip')
        ts_indi_df = ts_indi_df.join(agg_indi_by_user(ts_selector), how='inner')
        print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
        del in_selector, out_selector, ts_selector
        gc.collect()

    with timer('load classification(user) indi'):
        indi_type_id = 3
        indi_id = f'indi_{oof_seed}_{pred_type_id}_{indi_type_id}'
        indi_dir = f'gs-lgb-indi-train-{oof_seed}-{pred_type_id}-{indi_type_id}'

        in_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_ip'), compression='gzip').rename(
            columns={'pred': 'zp_prob'}).set_index(id_col)
        in_indi_df = in_indi_df.join(in_selector, how='inner')
        out_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_op'), compression='gzip').rename(
            columns={'pred': 'zp_prob'}).set_index(id_col)
        out_indi_df = out_indi_df.join(out_selector, how='inner')
        ts_selector = pd.read_pickle(os.path.join(root_dir, indi_dir, f'{indi_id}_p'), compression='gzip').rename(
            columns={'pred': 'zp_prob'}).set_index(id_col)
        ts_indi_df = ts_indi_df.join(ts_selector, how='inner')
        print(f'in_indi_df: {in_indi_df.shape}, out_indi_df: {out_indi_df.shape}, ts_indi_df: {ts_indi_df.shape}')
        del in_selector, out_selector, ts_selector
        gc.collect()

    cols = sorted(ts_indi_df.columns)
    return in_indi_df[cols], out_indi_df[cols], ts_indi_df[cols]


def get_oof_data(data_dir, in_indi_df, out_indi_df):
    with timer('get train src data'):
        train_df = pd.read_pickle(os.path.join(data_dir, 'train_df'), compression='gzip').loc[:, [id_col]]
        gc.collect()
        train_df_u = pd.read_pickle(os.path.join(data_dir, 'train_df_u'), compression='gzip').reset_index()
        tr_x_text = joblib.load(os.path.join(data_dir, 'tr_x_text_u'))
        tr_x_text = tr_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else tr_x_text
        print(train_df.shape, train_df_u.shape, tr_x_text.shape)

    with timer('oof split'):
        iind, oind = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=oof_seed * 1000000).split(
            train_df, groups=train_df[id_col]))
        ids = train_df.take(iind)[id_col]
        iind = train_df_u.loc[train_df_u[id_col].isin(ids)].index.values
        ids = train_df.take(oind)[id_col]
        oind = train_df_u.loc[train_df_u[id_col].isin(ids)].index.values
        train_df = train_df_u
        if need_prune_combo_cols:
            train_df = prune_combo_cols(train_df)
            gc.collect()

        in_df = train_df.take(iind)
        in_x_text = tr_x_text[iind]
        print(in_df.shape, in_x_text.shape)

        out_df = train_df.take(oind)
        out_x_text = tr_x_text[oind]
        print(out_df.shape, out_x_text.shape)
        del train_df, tr_x_text, train_df_u, iind, oind
        gc.collect()

    with timer('join indicators'):
        in_df = in_df.join(in_indi_df, on=id_col, how='inner')
        out_df = out_df.join(out_indi_df, on=id_col, how='inner')
        print(f'in_df: {in_df.shape}, out_df: {out_df.shape}')

    with timer('combine features'):
        in_selector = in_df.loc[:, ['fullVisitorId']].reset_index(drop=True)
        iy = np.log1p(in_df.target.values)
        ix = in_df.drop(['fullVisitorId', 'target'], axis=1).values
        ix = combine_features([ix, in_x_text]) if 'plain' != text_type else ix
        del in_df, in_x_text
        gc.collect()

        out_selector = out_df.loc[:, ['fullVisitorId']].reset_index(drop=True)
        oy = np.log1p(out_df.target.values)
        ox = out_df.drop(['fullVisitorId', 'target'], axis=1).values
        ox = combine_features([ox, out_x_text]) if 'plain' != text_type else ox
        del out_df, out_x_text
        gc.collect()

        print(ix.shape, ox.shape)

    return ix, iy, in_selector, ox, oy, out_selector


def get_test_data(data_dir, ts_indi_df):
    with timer('get test src data'):
        test_df = pd.read_pickle(os.path.join(data_dir, 'test_df_u'), compression='gzip').reset_index()
        if need_prune_combo_cols:
            test_df = prune_combo_cols(test_df)
            gc.collect()

        ts_x_text = joblib.load(os.path.join(data_dir, 'ts_x_text_u'))
        ts_x_text = ts_x_text.astype(np.bool).astype(np.float32) if 'bin' == text_type else ts_x_text
        print(test_df.shape, ts_x_text.shape)

    with timer('join indicators'):
        test_df = test_df.join(ts_indi_df, on=id_col, how='inner')
        print(f'test_df: {test_df.shape}')

    with timer('combine features'):
        ts_selector = test_df.loc[:, ['fullVisitorId']].reset_index(drop=True)
        ts_x = test_df.drop('fullVisitorId', axis=1).values
        gc.collect()
        ts_x = combine_features([ts_x, ts_x_text]) if 'plain' != text_type else ts_x
        print(ts_x.shape)
        del test_df, ts_x_text
        gc.collect()

    return ts_x, ts_selector


def learn(data, model_dir='.'):
    ix, iy, in_selector, ox, oy, out_selector = data

    with timer('train'):
        params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4,
                  'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 0.8,
                  'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
                  
        model = lgb.train(params, lgb.Dataset(ix, label=iy), 100000, [lgb.Dataset(ox, label=oy)],
                          early_stopping_rounds=200, verbose_eval=200)
        model_file_name = write_model(model, model_dir)
        print(f"{model_file_name}'s best iteration: {model.best_iteration}")

    validation(data, model, model_dir, model_file_name)

    return model, model_file_name


def validation(data, model, model_dir='.', model_file_name='lgb'):
    tx, ty, t_selector, ox, oy, out_selector = data

    with timer('validation'):
        tp = np.clip(model.predict(tx), 0, pred_max_val)
        del tx
        gc.collect()
        t_rmse = rmse(ty, tp)
        joblib.dump(tp, os.path.join(model_dir, f'{model_file_name}_tp'), compress=('gzip', 3))
        joblib.dump(ty, os.path.join(model_dir, f'{model_file_name}_ty'), compress=('gzip', 3))
        t_selector.to_pickle(f'{model_file_name}_tsr', compression='gzip')

        op = np.clip(model.predict(ox), 0, pred_max_val)
        del ox
        gc.collect()
        o_rmse = rmse(oy, op)
        joblib.dump(op, os.path.join(model_dir, f'{model_file_name}_op'), compress=('gzip', 3))
        joblib.dump(oy, os.path.join(model_dir, f'{model_file_name}_oy'), compress=('gzip', 3))
        out_selector.to_pickle(f'{model_file_name}_osr', compression='gzip')

        print(f't_rmse: {t_rmse}, o_rmse: {o_rmse}')


def run():
    # sys.stderr = sys.stdout = open(os.path.join('lgb_uv_submit_log'), 'w')

    print(os.listdir('../input'))
    data_dir = '../input/gs-get-src-data'
    print(os.listdir(data_dir))
    sub_data_dir = '../input/ga-customer-revenue-prediction'
    print(os.listdir(sub_data_dir))

    in_indi_df, out_indi_df, ts_indi_df = get_indicators()
    ix, iy, in_selector, ox, oy, out_selector = get_oof_data(data_dir, in_indi_df, out_indi_df)
    del in_indi_df, out_indi_df
    gc.collect()
    data = ix, iy, in_selector, ox, oy, out_selector
    model, model_file_name = learn(data)
    del ix, iy, in_selector, ox, oy, out_selector, data
    gc.collect()

    ts_x, ts_selector = get_test_data(data_dir, ts_indi_df)
    p = np.clip(model.predict(ts_x), 0, pred_max_val)
    submit(p, ts_selector[id_col].values, sub_data_dir, f'lgb_uv_{oof_seed}_{pred_type_id}')


if __name__ == '__main__':
    run()
