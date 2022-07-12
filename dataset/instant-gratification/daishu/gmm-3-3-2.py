import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import mixture
from scipy.stats import multivariate_normal

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

id_name = 'id'
label_name = 'target'
catFeature = 'wheezy-copper-turtle-magic'

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

first_gmm_valid_df = train_df[[id_name]]
first_gmm_valid_df[label_name] = 0
first_aug_valid_df = train_df[[id_name]]
first_aug_valid_df[label_name] = 0
gmm_add_valid_df = train_df[[id_name]]
gmm_add_valid_df[label_name] = 0
new_gmm_valid_df = train_df[[id_name]]
new_gmm_valid_df[label_name] = 0
norm_valid_df = train_df[[id_name]]
norm_valid_df[label_name] = 0
norm_submission_df = test_df[[id_name]]
norm_submission_df[label_name] = 0
trainThreshold = 0.15
testThreshold = 0.02
skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=888)

if norm_submission_df.shape[0] == 131073:
    norm_submission_df.to_csv('submission.csv',index=False)
    exit(0)

def QDA(trainX,trainY,validX,testX):
    model = QuadraticDiscriminantAnalysis(0.1)
    model.fit(trainX,trainY)
    train_pred_Y = model.predict_proba(trainX)[:,1]
    valid_pred_Y = model.predict_proba(validX)[:,1]
    test_pred_Y = model.predict_proba(testX)[:,1]
    return train_pred_Y,valid_pred_Y,test_pred_Y

def Get_pred(X,gauss0,gauss1):
    pred0 = np.array([m.pdf(X) for m in gauss0]).T
    pred1 = np.array([m.pdf(X) for m in gauss1]).T
    return np.max(pred1,axis=1) / (np.max(pred1,axis=1)+np.max(pred0,axis=1))

def Augment(x0,x1):
    init_u0 = np.mean(x0,axis=0)
    init_u1 = np.mean(x1,axis=0)
    u0 = init_u0 + 0.025 * (init_u0-init_u1)
    u1 = init_u1 + 0.025 * (init_u1-init_u0)
    aug_x0 = 2 * u0 - x0
    aug_x1 = 2 * u1 - x1
    return np.concatenate([x0,aug_x0]),np.concatenate([x1,aug_x1])

np.random.seed(817119)

for i in range(512):
    print('- train on magic = %s'%i)
    this_train_df = train_df.loc[train_df[catFeature]==i]
    this_test_df = test_df.loc[test_df[catFeature]==i]
    train_cols = [col for col in train_df.columns if (col not in [id_name,label_name,catFeature]) and (this_train_df[col].std() > 2)]
    train_X = this_train_df[train_cols].values
    train_Y = this_train_df[label_name].values
    test_X = this_test_df[train_cols].values
    first_gmm_valid_pred = np.zeros(len(train_X))
    first_aug_valid_pred = np.zeros(len(train_X))
    gmm_add_valid_pred = np.zeros(len(train_X))
    new_gmm_valid_pred = np.zeros(len(train_X))
    norm_valid_pred = np.zeros(len(train_X))
    norm_test_pred = np.zeros(len(test_X))
    for fold, (train_index, valid_index) in enumerate(skf.split(train_X,train_Y)):
        train_pred_Y,valid_pred_Y,test_pred_Y = QDA(train_X[train_index],train_Y[train_index],train_X[valid_index],test_X)
        revise_train_X = train_X[train_index]
        revise_train_Y = train_Y[train_index]
        revise_train_Y[(((train_Y[train_index].reshape(-1)==0)&(train_pred_Y.reshape(-1)>1-trainThreshold)))|(((train_Y[train_index].reshape(-1)==1)&(train_pred_Y.reshape(-1)<trainThreshold)))] = 1 - revise_train_Y[(((train_Y[train_index].reshape(-1)==0)&(train_pred_Y.reshape(-1)>1-trainThreshold)))|(((train_Y[train_index].reshape(-1)==1)&(train_pred_Y.reshape(-1)<trainThreshold)))]
        origin_test_X = test_X[(test_pred_Y<testThreshold)|(test_pred_Y>1-testThreshold)]
        origin_test_Y = test_pred_Y[(test_pred_Y<testThreshold)|(test_pred_Y>1-testThreshold)]
        origin_test_Y[origin_test_Y>0.5] = 1
        origin_test_Y[origin_test_Y<0.5] = 0
        x0 = np.concatenate([train_X[train_index][revise_train_Y==0],origin_test_X[origin_test_Y==0]])
        x1 = np.concatenate([train_X[train_index][revise_train_Y==1],origin_test_X[origin_test_Y==1]])
        gmm0 = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=5000, verbose=-1, verbose_interval=1)
        #gmm2.means_ = np.array([train_X[train_index][train_Y[train_index]==i].mean(axis=0) for i in range(2)])
        gmm0.fit(x0)
        gmm1 = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=5000, verbose=-1, verbose_interval=1)
        #gmm2.means_ = np.array([train_X[train_index][train_Y[train_index]==i].mean(axis=0) for i in range(2)])
        gmm1.fit(x1)
        mean0 = gmm0.means_
        cov0 = gmm0.covariances_
        multi_gauss0 = [multivariate_normal(mean=mean0[j], cov=cov0[j]) for j in range(3)]
        mean1 = gmm1.means_
        cov1 = gmm1.covariances_
        multi_gauss1 = [multivariate_normal(mean=mean1[j], cov=cov1[j]) for j in range(3)]
        valid_pred_Y = Get_pred(train_X[valid_index],multi_gauss0,multi_gauss1)
        #print('first pred',roc_auc_score(train_Y[valid_index],valid_pred_Y))
        first_gmm_valid_pred[valid_index] = Get_pred(train_X[valid_index],multi_gauss0,multi_gauss1)

        #
        train_pred_Y = Get_pred(train_X[train_index],multi_gauss0,multi_gauss1)
        test_pred_Y = Get_pred(test_X,multi_gauss0,multi_gauss1)
        revise_train_X = train_X[train_index]
        revise_train_Y = train_Y[train_index]
        revise_train_Y[(((train_Y[train_index].reshape(-1)==0)&(train_pred_Y.reshape(-1)>1-trainThreshold)))|(((train_Y[train_index].reshape(-1)==1)&(train_pred_Y.reshape(-1)<trainThreshold)))] = 1 - revise_train_Y[(((train_Y[train_index].reshape(-1)==0)&(train_pred_Y.reshape(-1)>1-trainThreshold)))|(((train_Y[train_index].reshape(-1)==1)&(train_pred_Y.reshape(-1)<trainThreshold)))]
        origin_test_X = test_X[(test_pred_Y<testThreshold)|(test_pred_Y>1-testThreshold)]
        origin_test_Y = test_pred_Y[(test_pred_Y<testThreshold)|(test_pred_Y>1-testThreshold)]
        origin_test_Y[origin_test_Y>0.5] = 1
        origin_test_Y[origin_test_Y<0.5] = 0
        x0 = np.concatenate([train_X[train_index][revise_train_Y==0],origin_test_X[origin_test_Y==0]])
        x1 = np.concatenate([train_X[train_index][revise_train_Y==1],origin_test_X[origin_test_Y==1]])
        gmm0 = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=5000, verbose=-1, verbose_interval=1)
        #gmm2.means_ = np.array([train_X[train_index][train_Y[train_index]==i].mean(axis=0) for i in range(2)])
        gmm0.fit(x0)
        gmm1 = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=5000, verbose=-1, verbose_interval=1)
        #gmm2.means_ = np.array([train_X[train_index][train_Y[train_index]==i].mean(axis=0) for i in range(2)])
        gmm1.fit(x1)
        mean0 = gmm0.means_
        cov0 = gmm0.covariances_
        multi_gauss0 = [multivariate_normal(mean=mean0[j], cov=cov0[j]) for j in range(3)]
        mean1 = gmm1.means_
        cov1 = gmm1.covariances_
        multi_gauss1 = [multivariate_normal(mean=mean1[j], cov=cov1[j]) for j in range(3)]
        gmm_add_valid_pred[valid_index] = Get_pred(train_X[valid_index],multi_gauss0,multi_gauss1)

        # new gmm
        mean0 = [(mean0[0]+mean0[1])/2.0,(mean0[1]+mean0[2])/2.0,(mean0[2]+mean0[0])/2.0]
        #cov0[np.abs(cov0)<0.5] = 0
        #p0 = np.stack([np.linalg.inv(cov) for cov in cov0])
        gmm0 = mixture.GaussianMixture(n_components=3, covariance_type='full', means_init=mean0, max_iter=5000, verbose=-1, verbose_interval=1)
        gmm0.fit(x0)
        mean0 = gmm0.means_
        cov0 = gmm0.covariances_
        multi_gauss0 = [multivariate_normal(mean=mean0[j], cov=cov0[j]) for j in range(3)]
        mean1 = [(mean1[0]+mean1[1])/2.0,(mean1[1]+mean1[2])/2.0,(mean1[2]+mean1[0])/2.0]
        #cov1[np.abs(cov1)<0.5] = 0
        #p1 = np.stack([np.linalg.inv(cov) for cov in cov1])
        gmm1 = mixture.GaussianMixture(n_components=3, covariance_type='full', means_init=mean1, max_iter=5000, verbose=-1, verbose_interval=1)
        gmm1.fit(x1)
        mean1 = gmm1.means_
        cov1 = gmm1.covariances_
        multi_gauss1 = [multivariate_normal(mean=mean1[j], cov=cov1[j]) for j in range(3)]
        new_gmm_valid_pred[valid_index] = Get_pred(train_X[valid_index],multi_gauss0,multi_gauss1)

        # aug
        aug_x0,aug_x1 = Augment(x0,x1)
        gmm0 = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=5000, verbose=-1, verbose_interval=1)
        #gmm2.means_ = np.array([train_X[train_index][train_Y[train_index]==i].mean(axis=0) for i in range(2)])
        gmm0.fit(aug_x0)
        gmm1 = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=5000, verbose=-1, verbose_interval=1)
        #gmm2.means_ = np.array([train_X[train_index][train_Y[train_index]==i].mean(axis=0) for i in range(2)])
        gmm1.fit(aug_x1)
        mean0 = gmm0.means_
        cov0 = gmm0.covariances_
        multi_gauss0 = [multivariate_normal(mean=mean0[j], cov=cov0[j]) for j in range(3)]
        mean1 = gmm1.means_
        cov1 = gmm1.covariances_
        multi_gauss1 = [multivariate_normal(mean=mean1[j], cov=cov1[j]) for j in range(3)]
        first_aug_valid_pred[valid_index] = Get_pred(train_X[valid_index],multi_gauss0,multi_gauss1)
        norm_valid_pred[valid_index] = Get_pred(train_X[valid_index],multi_gauss0,multi_gauss1)
        #print('second pred',roc_auc_score(train_Y[valid_index],norm_valid_pred[valid_index]))
        #print(roc_auc_score(train_Y[valid_index],valid_pred[valid_index]))
        #print(roc_auc_score(train_Y[valid_index],norm_valid_pred[valid_index]))
        pred0 = np.array([m.pdf(test_X) for m in multi_gauss0]).T
        pred1 = np.array([m.pdf(test_X) for m in multi_gauss1]).T
        norm_test_pred += np.max(pred1,axis=1) / (np.max(pred1,axis=1)+np.max(pred0,axis=1)) / 11.0

    print('first gmm',roc_auc_score(train_Y,first_gmm_valid_pred))
    print('gmm add',roc_auc_score(train_Y,gmm_add_valid_pred))
    print('new gmm',roc_auc_score(train_Y,new_gmm_valid_pred))
    print('aug gmm',roc_auc_score(train_Y,first_aug_valid_pred))
    first_gmm_valid_df.loc[this_train_df.index,label_name] = first_gmm_valid_pred
    first_aug_valid_df.loc[this_train_df.index,label_name] = first_aug_valid_pred
    gmm_add_valid_df.loc[this_train_df.index,label_name] = gmm_add_valid_pred
    new_gmm_valid_df.loc[this_train_df.index,label_name] = new_gmm_valid_pred
    norm_valid_df.loc[this_train_df.index,label_name] = norm_valid_pred
    norm_submission_df.loc[this_test_df.index,label_name] = norm_test_pred


def Save_df(validDf,prefix):
    metric = roc_auc_score(train_df['target'],validDf['target'])
    print(prefix,metric)
    validDf.to_csv('%s_valid_metric_%.6f.csv'%(prefix,metric),index=False)
    return None

Save_df(first_gmm_valid_df,'first_gmm')
Save_df(first_aug_valid_df,'first_aug')
Save_df(gmm_add_valid_df,'gmm_add')
Save_df(new_gmm_valid_df,'new_gmm')

metric = roc_auc_score(train_df['target'],norm_valid_df['target'])
norm_valid_df.to_csv('valid_metric_%.6f.csv'%(metric),index=False)
norm_submission_df.to_csv('submission_metric_%.6f.csv'%(metric),index=False)
norm_submission_df.to_csv('submission.csv',index=False)
