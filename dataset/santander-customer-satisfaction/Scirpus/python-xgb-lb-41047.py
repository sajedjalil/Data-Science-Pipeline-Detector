import gc
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

if __name__ == "__main__":
    print('Started!')
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    features = train.columns[1:-1]
    train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
    test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))

    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    features = train.columns[1:-1]
    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
    x_test_projected = pca.transform(normalize(test[features], axis=0))
    train.insert(1, 'PCAOne', x_train_projected[:, 0])
    train.insert(1, 'PCATwo', x_train_projected[:, 1])
    test.insert(1, 'PCAOne', x_test_projected[:, 0])
    test.insert(1, 'PCATwo', x_test_projected[:, 1])
    tokeep = ['num_var39_0',  # 0.00031104199066874026
              'ind_var13',  # 0.00031104199066874026
              'num_op_var41_comer_ult3',  # 0.00031104199066874026
              'num_var43_recib_ult1',  # 0.00031104199066874026
              'imp_op_var41_comer_ult3',  # 0.00031104199066874026
              'num_var8',  # 0.00031104199066874026
              'num_var42',  # 0.00031104199066874026
              'num_var30',  # 0.00031104199066874026
              'saldo_var8',  # 0.00031104199066874026
              'num_op_var39_efect_ult3',  # 0.00031104199066874026
              'num_op_var39_comer_ult3',  # 0.00031104199066874026
              'num_var41_0',  # 0.0006220839813374805
              'num_op_var39_ult3',  # 0.0006220839813374805
              'saldo_var13',  # 0.0009331259720062209
              'num_var30_0',  # 0.0009331259720062209
              'ind_var37_cte',  # 0.0009331259720062209
              'ind_var39_0',  # 0.001244167962674961
              'num_var5',  # 0.0015552099533437014
              'ind_var10_ult1',  # 0.0015552099533437014
              'num_op_var39_hace2',  # 0.0018662519440124418
              'num_var22_hace2',  # 0.0018662519440124418
              'num_var35',  # 0.0018662519440124418
              'ind_var30',  # 0.0018662519440124418
              'num_med_var22_ult3',  # 0.002177293934681182
              'imp_op_var41_efect_ult1',  # 0.002488335925349922
              'var36',  # 0.0027993779160186624
              'num_med_var45_ult3',  # 0.003110419906687403
              'imp_op_var39_ult1',  # 0.0037325038880248835
              'imp_op_var39_comer_ult3',  # 0.0037325038880248835
              'imp_trans_var37_ult1',  # 0.004043545878693624
              'num_var5_0',  # 0.004043545878693624
              'num_var45_ult1',  # 0.004665629860031105
              'ind_var41_0',  # 0.0052877138413685845
              'imp_op_var41_ult1',  # 0.0052877138413685845
              'num_var8_0',  # 0.005598755832037325
              'imp_op_var41_efect_ult3',  # 0.007153965785381027
              'num_op_var41_ult3',  # 0.007153965785381027
              'num_var22_hace3',  # 0.008087091757387248
              'num_var4',  # 0.008087091757387248
              'imp_op_var39_comer_ult1',  # 0.008398133748055987
              'num_var45_ult3',  # 0.008709175738724729
              'ind_var5',  # 0.009953343701399688
              'imp_op_var39_efect_ult3',  # 0.009953343701399688
              'num_meses_var5_ult3',  # 0.009953343701399688
              'saldo_var42',  # 0.01181959564541213
              'imp_op_var39_efect_ult1',  # 0.013374805598755831
              'PCATwo',  # 0.013996889580093312
              'num_var45_hace2',  # 0.014618973561430793
              'num_var22_ult1',  # 0.017107309486780714
              'saldo_medio_var5_ult1',  # 0.017418351477449457
              'PCAOne',  # 0.018040435458786936
              'saldo_var5',  # 0.0208398133748056
              'ind_var8_0',  # 0.021150855365474338
              'ind_var5_0',  # 0.02177293934681182
              'num_meses_var39_vig_ult3',  # 0.024572317262830483
              'saldo_medio_var5_ult3',  # 0.024883359253499222
              'num_var45_hace3',  # 0.026749611197511663
              'num_var22_ult3',  # 0.03452566096423017
              'saldo_medio_var5_hace3',  # 0.04074650077760498
              'saldo_medio_var5_hace2',  # 0.04292379471228616
              'SumZeros',  # 0.04696734059097978
              'saldo_var30',  # 0.09611197511664074
              'var38',  # 0.1390357698289269
              'var15']  # 0.20964230171073095
    features = train.columns[1:-1]
    todrop = list(set(tokeep).difference(set(features)))
    train.drop(todrop, inplace=True, axis=1)
    test.drop(todrop, inplace=True, axis=1)
    features = train.columns[1:-1]
    split = 10
    skf = StratifiedKFold(train.TARGET.values,
                          n_folds=split,
                          shuffle=False,
                          random_state=42)

    train_preds = None
    test_preds = None
    visibletrain = blindtrain = train
    index = 0
    print('Change num_rounds to 350')
    num_rounds = 10
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.03
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["max_depth"] = 5
    params["min_child_weight"] = 1
    params["eval_metric"] = "auc"
    for train_index, test_index in skf:
        print('Fold:', index)
        visibletrain = train.iloc[train_index]
        blindtrain = train.iloc[test_index]
        dvisibletrain = \
            xgb.DMatrix(csr_matrix(visibletrain[features]),
                        visibletrain.TARGET.values,
                        silent=True)
        dblindtrain = \
            xgb.DMatrix(csr_matrix(blindtrain[features]),
                        blindtrain.TARGET.values,
                        silent=True)
        watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
        clf = xgb.train(params, dvisibletrain, num_rounds,
                        evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=False)

        blind_preds = clf.predict(dblindtrain)
        print('Blind Log Loss:', log_loss(blindtrain.TARGET.values,
                                          blind_preds))
        print('Blind ROC:', roc_auc_score(blindtrain.TARGET.values,
                                          blind_preds))
        index = index+1
        del visibletrain
        del blindtrain
        del dvisibletrain
        del dblindtrain
        gc.collect()
        dfulltrain = \
            xgb.DMatrix(csr_matrix(train[features]),
                        train.TARGET.values,
                        silent=True)
        dfulltest = \
            xgb.DMatrix(csr_matrix(test[features]),
                        silent=True)
        if(train_preds is None):
            train_preds = clf.predict(dfulltrain)
            test_preds = clf.predict(dfulltest)
        else:
            train_preds *= clf.predict(dfulltrain)
            test_preds *= clf.predict(dfulltest)
        del dfulltrain
        del dfulltest
        del clf
        gc.collect()

    train_preds = np.power(train_preds, 1./index)
    test_preds = np.power(test_preds, 1./index)
    print('Average Log Loss:', log_loss(train.TARGET.values, train_preds))
    print('Average ROC:', roc_auc_score(train.TARGET.values, train_preds))
    submission = pd.DataFrame({"ID": train.ID,
                               "TARGET": train.TARGET,
                               "PREDICTION": train_preds})

    submission.to_csv("simplexgbtrain.csv", index=False)
    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_preds})
    submission.to_csv("simplexgbtest.csv", index=False)
    print('Finish')
