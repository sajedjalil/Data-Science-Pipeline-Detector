import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score



nineteenfeatures = [#'saldo_medio_var5_hace2',
                    #'num_meses_var39_vig_ult3',
                    #'num_var42',
                    
                    'num_var5',
                    #'saldo_medio_var5_hace2',
                    #'saldo_var42',
                    #'num_var45_hace2'
    
                    #'saldo_var30',
                    #'var15',
                    #'saldo_var5',
                    #'ind_var30',
                    #'var38',
                    #'saldo_medio_var5_ult3',
                    #'num_meses_var5_ult3'
                    #'var36',
                    #'num_meses_var39_vig_ult3'
                    ]
if __name__ == "__main__":
    print('Started!')
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    num_rounds = 50 #500
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.1 #0.02
    params["subsample"] = 0.55
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["max_depth"] = 2
    params["eval_metric"] = "auc"
    params["seed"] = 1234567 #0

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
    submission.to_csv("xgb_19Fea_v70_Apr11.csv", index=False)
    print('Completed!')