import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

#nineteenfeatures = ['imp_ent_var16_ult1',
#                    'var38',
#                   'ind_var30',
 #                   'delta_imp_aport_var13_1y3',
  #                  'saldo_medio_var13_corto_hace2',
   #                 'num_op_var39_hace3',
    #                'imp_var43_emit_ult1',
     #               'num_meses_var5_ult3',
      #              'delta_num_aport_var13_1y3',
       #             'num_var42_0',
        #            'imp_op_var40_ult1',
         #           'num_var22_ult1',
          #          'saldo_var5',
           #         'num_op_var40_ult1',
            #        'imp_aport_var13_ult1',
             #       'saldo_var42', 'ind_var39_0',
              #      'num_aport_var13_ult1',
               #     'var15']
               
##To be accurate, top10 features               
top10features = ['var15',                 #0.70             250 rounds
                     'saldo_var30',           #0.812096189048   250 rounds
                     'std',                   #0.816972944498   250 rounds
                     'num_var22_ult3',        #0.829681738232   325 rounds
                     'imp_op_var39_ult1',     #0.833324036977   325 rounds
                     'num_var45_hace3',       #0.8347158495     325 rounds
                     'saldo_medio_var5_hace2',#0.836754399288   325 rounds
                     'var3',                  #0.838416590074   325 rounds
                     'saldo_medio_var8_ult3',
                     'ind_var41_0'            #0.836755316781 - 0.839971060754   325 rounds
]

if __name__ == "__main__":
    print('Started!')
    np.random.seed(8)

    print('Load data...')
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    ntrain = train.shape[0]
    target = train['TARGET']
    ID_test = test['ID']

    train = train.drop(['ID','TARGET'], axis=1)
    test = test.drop('ID', axis=1)

    # New features
    print('Computing new features...')
    train_test = pd.concat((train, test), axis=0)
    features = train_test.columns
    train_test['std'] = train_test.apply(lambda x: np.std(x), axis=1)


    tr = train_test.iloc[:ntrain, :]
    te = train_test.iloc[ntrain:, :]


    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    df_train['std']=tr['std'].values
    df_test['std']=te['std'].values
    
    num_rounds = 50 #500
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.1 #0.02
    params["subsample"] = 0.6
    params["colsample_bytree"] = 0.6
    params["silent"] = 1
    params["max_depth"] = 2
    params["eval_metric"] = "auc"
    params["seed"] = 8888 #0
    
    
    dtrain = xgb.DMatrix(df_train[top10features],
                         df_train.TARGET.values,
                         silent=True)
    dtest = xgb.DMatrix(df_test[top10features],
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
    submission.to_csv("xgb_top10Fea_v1.csv", index=False)
    print('Completed!')