import kagglegym
import numpy as np
from sklearn import linear_model as lm
from sklearn.metrics import r2_score
import pandas as pd
pd.set_option('chained_assignment',None)

low_y_cut   = -0.086093
high_y_cut  =  0.093497


env             = kagglegym.make()
observation     = env.reset()
train           = observation.train
y_is_above_cut  = (train.y > high_y_cut)
y_is_below_cut  = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
train           = train.loc[y_is_within_cut,:]
train           = train[["id", "y", "timestamp","technical_20","technical_13","technical_30"]]

train["technical_20_shifted"]     = train.groupby('id')["technical_20"].shift(1)
train["technical_13_shifted"]     = train.groupby('id')["technical_13"].shift(1)
train["technical_30_shifted"]     = train.groupby('id')["technical_30"].shift(1)
train["y_shifted"]                = train.groupby('id')["y"].shift(1)
train['derived_feat']             = ( (train["technical_20"]         + train["technical_13"]         - train["technical_30"])  \
                                    -0.9327 * (train["technical_20_shifted"] + train["technical_13_shifted"] - train["technical_30_shifted"]) \
                                    )/(1 - 0.9327)

for col in train.columns:
    train.loc[(np.isfinite(train[col]) == 0),:] = 0

fit_lr_bound    = np.min(train['timestamp']) + 50
model           = lm.LinearRegression()

train_fit       = train[train['timestamp']<fit_lr_bound]


model.fit(train_fit['derived_feat'].reshape(len(train_fit),1),train_fit['y'])

coefs = [.075,1.]

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    return r

def fit_derived_coeffs(reward,preds,a1,a2,b1,b2):
    fit_alpha,fit_beta = 1e6,1e6
    residual  = 1e6
    for beta in [-1.,0,1.]:
        t1 = a1 + beta*b1
        t2 = a2 + beta*b2
        for alpha in np.linspace(0.01,.25,25):
                y = ((t1  - (1 - alpha) *t2)/alpha)
                r_score_fit = r_score(y,preds)
                temp_residual = np.abs(r_score_fit - reward)/np.abs(reward)
                if (temp_residual < residual ):
                    residual = temp_residual
                    fit_alpha = alpha
                    fit_beta = beta
    return ([fit_alpha,fit_beta],residual)

def update_derived_coeffs(coefs,fits_residual,lr = .125):
    fits     = fits_residual [0]
    residual = fits_residual[1]
    coefs[0] = coefs[0] + lr*(fits[0] - coefs[0]) * max(1 - residual,0)
    coefs[1] = coefs[1] + lr*(fits[1] - coefs[1]) * max(1 - residual,0)
    return coefs,residual

def update_lr_coeff(residual,actions,yhat_reco,var_to_update,coeff,lr = .125):
    dphi    =  (lr * np.mean( (yhat_reco - actions)*var_to_update)) * max((1 - residual),0)
    return coeff + dphi

def convert(actions,ids):
    d_ = {}
    for id,ele in enumerate(ids):
        d_[ele[0]] = actions[id]

    y_ = []
    for ele in observation.target.id:
        try:
            val = d_[ele]
            y_.append(val)
        except:
            y_.append(0)
    return y_




for i in range(fit_lr_bound+1,np.max(train['timestamp'].unique())-2):
    subset = train[train['timestamp']== i]
    subset_future = train[train['timestamp'] == i+1]
    subset['derived_feat_better'] = ( (subset["technical_20"]         + coefs[1]*subset["technical_13"]         - subset["technical_30"])  \
                                    - (1 - coefs[0]) * (subset["technical_20_shifted"] + coefs[1]*subset["technical_13_shifted"] - subset["technical_30_shifted"]))/coefs[0]
    actions = model.predict(subset['derived_feat_better'].reshape(len(subset),1)).clip(low_y_cut,high_y_cut)
    reward  = r_score(subset['y'], actions)
    coefs,residual = update_derived_coeffs(coefs, fit_derived_coeffs(reward,actions,
                                (subset["technical_20"] - subset["technical_30"]).values,
                                (subset["technical_20_shifted"] - subset["technical_30_shifted"]).values,
                                subset["technical_13"].values,subset["technical_13_shifted"].values))
    subset_future['derived_feat_better'] = ( (subset_future["technical_20"]         + coefs[1]*subset_future["technical_13"]     - subset_future["technical_30"])  \
                                    - (1 - coefs[0]) * (subset_future["technical_20_shifted"] + coefs[1]*subset_future["technical_13_shifted"] - subset_future["technical_30_shifted"]))/coefs[0]
    subset['actions'] = actions

    subset_future = subset_future.merge(subset,on='id',suffixes=('','_shifted_2'))

    for col in ['actions','derived_feat_better']:
        subset_future   = subset_future.loc[np.isfinite(subset_future[col]),:]
    model.coef_[0] = update_lr_coeff(reward,subset_future['actions'],subset_future['derived_feat_better'],subset_future['derived_feat_better_shifted_2'],model.coef_[0])

print("Model Trained.")

reward,residual = 1e6,1e6
done = False
actions,prev_feats,prev_feats_derived_feat = [],[],[]

while True:
    if done:
        break

    features      = observation.features
    features      = features[["id", "timestamp","technical_20","technical_30","technical_13"]]
    features_copy = features.copy()

    for col in features.columns:
        features.loc[(np.isfinite(features[col]) == 0),:] = 0

    if (len(prev_feats) == 0):
        prev_feats              = features_copy
        prev_feats["preds"]     = np.zeros(len(prev_feats))
        observation.target.y    = np.zeros(len(observation.target.id))
        target                  = observation.target
        observation, reward, done, info \
                                = env.step(target)
        continue


    features = features.merge(prev_feats[["id", "timestamp","technical_20","technical_30","technical_13","preds"]],on='id',suffixes=('','_shifted'))
    features = features.dropna()

    if (reward != 1e6):
        coefs,residual = update_derived_coeffs(coefs, fit_derived_coeffs(reward,features['preds'],
                            (features["technical_20"] - features["technical_30"]).values,
                            (features["technical_20_shifted"] - features["technical_30_shifted"]).values,
                            features["technical_13"].values,features["technical_13_shifted"].values))
    features['derived_feat'] \
             = ( (features["technical_20"]           + coefs[1]*features["technical_13"]          - features["technical_30"])  \
                - (1-coefs[0]) * (features["technical_20_shifted"] + coefs[1]*features["technical_13_shifted"]  - features["technical_30_shifted"]))/coefs[0]
    derived_features_copy = features.copy()

    if (len(features) > 0  ):
        actions = model.predict(features['derived_feat'].reshape(len(features),1)).clip(low_y_cut,high_y_cut)
        observation.target.y = convert(actions,features[['id']].as_matrix())

        if (reward != 1e6 and len(actions) != 0 and residual != 1e6):
            if (len(prev_feats_derived_feat) != 0):
                features = features.merge(prev_feats_derived_feat[['id','derived_feat']],on='id',suffixes=('','_shifted'))
                features.dropna()
                model.coef_[0] = update_lr_coeff(residual,features['preds'],features['derived_feat'],features['derived_feat_shifted'],model.coef_[0])


    else:
        prev_feats              = features_copy
        prev_feats["preds"]     = np.zeros(len(prev_feats))
        prev_feats_derived_feat = prev_feats.merge(derived_features_copy,on='id',suffixes=('','_shifted'))
        observation.target.y    = np.zeros(len(observation.target.id))
        target                  = observation.target
        observation, reward, done, info\
                                = env.step(target)
        continue

    if len(observation.features < 0):
        timestamp = observation.features["timestamp"][0]
        if timestamp % 100 == 0:
            print("Timestamp #{}".format(timestamp))

    prev_feats                      = features_copy
    prev_feats['preds']             = observation.target.y
    prev_feats_derived_feat         = prev_feats.merge(derived_features_copy,on='id',suffixes=('','_shifted'))
    observation, reward, done, info = env.step(observation.target)

print(info)
