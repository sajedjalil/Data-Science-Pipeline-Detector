import pandas as p; from sklearn import *
import warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count

t, r = [p.read_csv('../input/' + f) for f in ['train.csv', 'test.csv']]
cl = 'wheezy-copper-turtle-magic'; re_ = []
col = [c for c in t.columns if c not in ['id', 'target', cl]]

def multi_pred(s, t_, r_):
    sv1 = svm.NuSVC(probability=True, kernel='poly', degree=4, random_state=4, nu=0.59, coef0=0.053)
    sv2 = discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=0.112)
    df = p.concat((t_, r_)).reset_index(drop=True)
    df2 = preprocessing.StandardScaler().fit_transform(decomposition.PCA(n_components=40, random_state=4).fit_transform(df[col]))
    df3 = preprocessing.StandardScaler().fit_transform(decomposition.TruncatedSVD(n_components=40, random_state=4).fit_transform(df[col]))
    df = p.concat((df, p.DataFrame(df2), p.DataFrame(df3)), axis=1)
    t_ = df[:t_.shape[0]]
    r_ = df[t_.shape[0]:]
    col2 = [c for c in t_.columns if c not in ['id', 'target', cl] + col]
    x1, x2, y1, y2 = model_selection.train_test_split(t_[col2], t_['target'], test_size=0.2, random_state=99)
    sv1.fit(x1, y1)
    sv2.fit(x1, y1)
    score1 = metrics.roc_auc_score(y2, sv1.predict_proba(x2)[:,1])
    score2 = metrics.roc_auc_score(y2, sv2.predict_proba(x2)[:,1])
    sv1.fit(t_[col2], t_['target'])
    sv2.fit(t_[col2], t_['target'])
    if score1 >= score2:
        r_['target'] = ((sv1.predict_proba(r_[col2])[:,1] * 0.54) + (sv2.predict_proba(r_[col2])[:,1] * 0.46))
    else:
        r_['target'] = ((sv1.predict_proba(r_[col2])[:,1] * 0.46) + (sv2.predict_proba(r_[col2])[:,1] * 0.54))
    
    #feedback overfit loop
    r_2 = r_[((r_.target > 0.84) | (r_.target < 0.16))]
    r_2['target'] = r_2['target'].map(lambda x: 1 if x > 0.5 else 0)
    t_ = p.concat([t_, r_2]).reset_index(drop=True)
    sv1.fit(t_[col2], t_['target'])
    sv2.fit(t_[col2], t_['target'])
    r_['target'] = ((sv1.predict_proba(r_[col2])[:,1] * 0.5) + (sv2.predict_proba(r_[col2])[:,1] * 0.5))
    return r_

p_ = Pool(cpu_count())
sections = [[s, t[t[cl]==s], r[r[cl]==s]] for s in sorted(t[cl].unique())]
re_ = p_.starmap(multi_pred, sections)
p_.close(); p_.join();
preds = p.concat(re_)[['id','target']]
mm = preprocessing.MinMaxScaler(feature_range=(0.45, 0.55))
preds['target'] = mm.fit_transform(preds['target'].values.reshape(-1, 1))
preds.to_csv("submission.csv", index=False)