import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score


nineteenfeatures = ['imp_ent_var16_ult1',
                    'var38',
                    'ind_var30',
                    'delta_imp_aport_var13_1y3',
                    'saldo_medio_var13_corto_hace2',
                    'num_op_var39_hace3',
                    'imp_var43_emit_ult1',
                    'num_meses_var5_ult3',
                    'delta_num_aport_var13_1y3',
                    'num_var42_0',
                    'imp_op_var40_ult1',
                    'num_var22_ult1',
                    'saldo_var5',
                    'num_op_var40_ult1',
                    'imp_aport_var13_ult1',
                    'saldo_var42', 'ind_var39_0',
                    'num_aport_var13_ult1',
                    'var15']

if __name__ == "__main__":
    print('Started!')
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    num_rounds = 500
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.02
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["max_depth"] = 6
    params["eval_metric"] = "auc"

    dtrain = xgb.DMatrix(df_train[nineteenfeatures],
                         df_train.TARGET.values,
                         silent=True)
    dtest = xgb.DMatrix(df_test[nineteenfeatures],
                        silent=True)

    clf = xgb.train(params, dtrain, num_rounds,
                    verbose_eval=True)

    y_pred = clf.predict(dtrain)
    print('Log Loss:', log_loss(df_train.TARGET.values, y_pred))
    print('ROC:', roc_auc_score(df_train.TARGET.values, y_pred))
    y_pred = clf.predict(dtest)
    submission = pd.DataFrame({"ID": df_test.ID, "TARGET": y_pred})
    submission.fillna((df_train.TARGET == 1).sum() /
                      (df_train.TARGET == 0).sum(),
                      inplace=True)
    submission.to_csv("xgbsubmission.csv", index=False)
    print('Completed!')