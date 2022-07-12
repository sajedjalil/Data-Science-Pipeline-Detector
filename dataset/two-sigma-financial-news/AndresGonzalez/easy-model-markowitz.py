from sklearn.covariance import ledoit_wolf
import cvxpy as cvx
import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(train, _) = env.get_training_data()
VARSEL = 'returnsOpenPrevMktres10'


def max_ret_min_var(mu, B, b=0.01):
    n = mu.shape[0]
    x = cvx.Variable(n)
    ret = cvx.sum(x*mu)
    risk = cvx.quad_form(x, cvx.Parameter(shape=B.shape, value=B, PSD=True))
    objective = cvx.Minimize(-ret + b*risk)
    constraint = [x >= -1, x <= 1, cvx.sum(x) == 0,
                  cvx.sum(cvx.abs(x)) <= 1]
    Problem = cvx.Problem(objective, constraint)
    Problem.solve()
    return x.value

def min_var_fix_ret(mu, B, obj=0.01, minRisk=True):
    n = mu.shape[0]
    x = cvx.Variable(n)
    ret = cvx.sum(x*mu)
    risk = cvx.quad_form(x, cvx.Parameter(shape=B.shape, value=B, PSD=True))
    if minRisk == True:
        objective = cvx.Minimize(risk)
        constraint = [x >= -1, x <= 1, ret >= obj]
    else:
        objective = cvx.Maximize(ret)
        constraint = [x >= -1, x <= 1, risk <= obj]
    Problem = cvx.Problem(objective, constraint)
    Problem.solve()
    return x.value


def ledoit_wolf_cov(r):
    P = ledoit_wolf(r.T)[0]
    P = pd.DataFrame(P)
    P.index = r.index
    P.columns = r.index
    return P

r = train.pivot(index='assetCode', columns='time',
                values=VARSEL).fillna(0)
risk_m = ledoit_wolf_cov(r)

days = env.get_prediction_days()
market_fcst = pd.DataFrame()


for market, news, preds in days:
    market['fcst_mk'] = market[VARSEL].fillna(0)
    market['fcst_mk2'] = (market.fcst_mk - np.mean(market.fcst_mk))/\
                         sum(abs(market.fcst_mk))
    market_fcst = pd.concat([market_fcst,
                             market[['time', 'assetCode', 'fcst_mk2']]])
    fcst_smooth = market_fcst.groupby('assetCode').\
                  apply(lambda x: np.mean(x['fcst_mk2'].tail(5))).reset_index()
    fcst_smooth.columns.values[1] = 'fcst_mk4'
    market = market.merge(fcst_smooth, how='left')
    ## markowitz
    B = market[['assetCode', 'fcst_mk4']].merge(risk_m.reset_index(), how = 'inner')
    assets = B[['assetCode', 'fcst_mk4']]
    B = B[B.assetCode].values
    x = min_var_fix_ret(assets.fcst_mk4, B, 0.02)
    #x = max_ret_min_var(assets.fcst_mk4, B,100)
    #assets['fcst_mks'] = x/sum(abs(x))
    assets['fcst_mks'] = x
    market = market.merge(assets[['assetCode', 'fcst_mks']], how='left')
    preds = preds.merge(market[['assetCode', 'fcst_mks']], how='left')
    preds = preds.drop(columns='confidenceValue').rename(columns={'fcst_mks' : 'confidenceValue'})
    preds.confidenceValue.fillna(0, inplace=True)
    env.predict(preds)

env.write_submission_file()