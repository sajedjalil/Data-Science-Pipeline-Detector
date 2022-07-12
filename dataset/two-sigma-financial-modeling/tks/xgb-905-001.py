import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

env = kagglegym.make()
o = env.reset()

#train = pd.read_hdf('../input/train.h5')
y_train = o.train['y']

cols_lst = [['technical_20', 'technical_27', 'technical_7', 'technical_35',
             'technical_30', 'technical_40', 'technical_36', 'fundamental_53',
             'technical_21', 'technical_2', 'fundamental_56', 'fundamental_14',
             'fundamental_15', 'technical_17', 'fundamental_45',
             'fundamental_39', 'technical_19', 'fundamental_17',
             'fundamental_8', 'fundamental_42', 'fundamental_5',
             'fundamental_41', 'technical_43', 'fundamental_7',
             'fundamental_43', 'fundamental_13', 'fundamental_60',
             'fundamental_18', 'fundamental_23', 'technical_12', 'technical_22',
             'fundamental_2', 'technical_14', 'fundamental_58', 'technical_11',
             'fundamental_46', 'fundamental_59', 'fundamental_10',
             'fundamental_21', 'fundamental_54'],
            ['technical_20', 'technical_7', 'technical_27', 'technical_35',
             'technical_30', 'technical_40', 'technical_36', 'fundamental_53',
             'technical_17', 'fundamental_56', 'technical_21', 'technical_19',
             'fundamental_42', 'fundamental_15', 'fundamental_39',
             'fundamental_8', 'fundamental_41', 'fundamental_45',
             'fundamental_17', 'technical_2', 'fundamental_7', 'fundamental_5',
             'fundamental_18', 'fundamental_13', 'technical_22',
             'fundamental_46', 'technical_43', 'fundamental_60',
             'fundamental_14', 'technical_14', 'derived_1', 'fundamental_20',
             'fundamental_23', 'fundamental_44', 'fundamental_10',
             'fundamental_61', 'technical_12', 'fundamental_11',
             'fundamental_30', 'fundamental_57'],
            ['technical_20', 'technical_27', 'technical_35', 'technical_7',
             'technical_30', 'technical_40', 'fundamental_53', 'technical_21',
             'technical_19', 'technical_36', 'fundamental_42', 'technical_17',
             'fundamental_56', 'technical_2', 'fundamental_14',
             'fundamental_15', 'fundamental_39', 'fundamental_8',
             'fundamental_45', 'fundamental_5', 'fundamental_41',
             'technical_43', 'fundamental_7', 'fundamental_62',
             'fundamental_17', 'fundamental_20', 'fundamental_23',
             'fundamental_60', 'fundamental_21', 'derived_1', 'fundamental_18',
             'fundamental_36', 'fundamental_43', 'fundamental_50',
             'technical_14', 'technical_29', 'fundamental_26', 'fundamental_59',
             'technical_12', 'technical_22'],
            ['technical_20', 'technical_27', 'technical_7', 'technical_35',
             'technical_30', 'technical_36', 'technical_40', 'fundamental_53',
             'fundamental_42', 'fundamental_15', 'fundamental_56',
             'technical_21', 'technical_19', 'technical_17', 'fundamental_39',
             'fundamental_8', 'technical_2', 'fundamental_46', 'fundamental_45',
             'fundamental_7', 'fundamental_41', 'fundamental_14',
             'fundamental_43', 'fundamental_17', 'fundamental_50',
             'fundamental_36', 'derived_1', 'technical_14', 'technical_43',
             'fundamental_18', 'fundamental_21', 'fundamental_54',
             'fundamental_23', 'fundamental_20', 'fundamental_13',
             'fundamental_5', 'fundamental_44', 'technical_29',
             'fundamental_10', 'technical_12'],
            ['technical_20', 'technical_27', 'technical_7', 'technical_35',
             'technical_30', 'technical_40', 'technical_36', 'fundamental_53',
             'fundamental_14', 'technical_21', 'technical_19', 'fundamental_15',
             'technical_17', 'fundamental_56', 'technical_2', 'fundamental_45',
             'fundamental_8', 'fundamental_42', 'fundamental_39', 'derived_1',
             'fundamental_7', 'fundamental_18', 'fundamental_46',
             'fundamental_60', 'fundamental_16', 'technical_29',
             'fundamental_43', 'fundamental_5', 'fundamental_62',
             'fundamental_17', 'fundamental_41', 'technical_22', 'technical_12',
             'fundamental_20', 'technical_14', 'fundamental_10', 'technical_43',
             'fundamental_50', 'fundamental_57', 'technical_6'],
            ['technical_20', 'technical_27', 'technical_7', 'technical_35',
             'technical_30', 'technical_36', 'fundamental_53', 'technical_40',
             'technical_19', 'fundamental_56', 'fundamental_14',
             'fundamental_42', 'technical_17', 'technical_21', 'fundamental_15',
             'fundamental_41', 'fundamental_39', 'fundamental_8', 'technical_2',
             'fundamental_45', 'derived_1', 'fundamental_5', 'fundamental_17',
             'fundamental_46', 'fundamental_60', 'fundamental_13',
             'fundamental_43', 'fundamental_18', 'fundamental_23',
             'technical_22', 'technical_29', 'fundamental_7', 'fundamental_20',
             'fundamental_21', 'fundamental_50', 'technical_43',
             'fundamental_27', 'technical_11', 'technical_14', 'technical_12'],
            ['technical_20', 'technical_7', 'technical_27', 'technical_35',
             'technical_36', 'technical_30', 'technical_19', 'fundamental_53',
             'technical_40', 'technical_21', 'fundamental_56', 'fundamental_41',
             'fundamental_15', 'fundamental_45', 'fundamental_42',
             'technical_17', 'fundamental_5', 'fundamental_8', 'fundamental_20',
             'fundamental_46', 'fundamental_7', 'technical_2', 'fundamental_14',
             'fundamental_43', 'fundamental_39', 'derived_1', 'fundamental_60',
             'fundamental_10', 'fundamental_23', 'fundamental_13',
             'technical_43', 'fundamental_17', 'technical_29', 'fundamental_21',
             'fundamental_16', 'technical_11', 'fundamental_18',
             'fundamental_61', 'fundamental_2', 'fundamental_47'],
            ['technical_20', 'technical_27', 'technical_7', 'technical_35',
             'technical_36', 'technical_30', 'technical_40', 'fundamental_53',
             'fundamental_45', 'technical_19', 'fundamental_42', 'technical_17',
             'fundamental_39', 'fundamental_56', 'technical_21',
             'fundamental_7', 'technical_2', 'fundamental_8', 'fundamental_17',
             'fundamental_15', 'fundamental_14', 'fundamental_41',
             'fundamental_60', 'fundamental_18', 'fundamental_5',
             'fundamental_46', 'fundamental_43', 'technical_29',
             'fundamental_21', 'fundamental_61', 'technical_14',
             'fundamental_50', 'fundamental_10', 'fundamental_16',
             'fundamental_35', 'derived_1', 'technical_43', 'technical_12',
             'fundamental_13', 'fundamental_59'],
            ['technical_20', 'technical_27', 'technical_7', 'technical_35',
             'technical_36', 'technical_30', 'technical_40', 'fundamental_53',
             'technical_2', 'fundamental_42', 'fundamental_41',
             'fundamental_56', 'technical_21', 'fundamental_15', 'technical_19',
             'technical_17', 'fundamental_46', 'fundamental_5',
             'fundamental_20', 'fundamental_7', 'fundamental_8',
             'fundamental_45', 'fundamental_39', 'fundamental_14',
             'fundamental_17', 'technical_22', 'technical_43', 'fundamental_43',
             'fundamental_10', 'derived_1', 'fundamental_60', 'fundamental_21',
             'fundamental_18', 'technical_29', 'fundamental_16', 'technical_12',
             'fundamental_59', 'fundamental_50', 'derived_3', 'fundamental_2'],
            ['technical_20', 'technical_7', 'technical_27', 'technical_36',
             'technical_35', 'technical_30', 'technical_40', 'fundamental_39',
             'fundamental_53', 'fundamental_42', 'technical_19', 'technical_2',
             'fundamental_45', 'technical_17', 'fundamental_15', 'technical_21',
             'fundamental_46', 'fundamental_56', 'fundamental_8', 'derived_1',
             'fundamental_41', 'technical_22', 'fundamental_14',
             'fundamental_60', 'fundamental_23', 'fundamental_17',
             'fundamental_18', 'fundamental_10', 'fundamental_5',
             'technical_43', 'fundamental_62', 'fundamental_43',
             'fundamental_36', 'fundamental_20', 'fundamental_61',
             'fundamental_7', 'fundamental_50', 'fundamental_59',
             'technical_14', 'fundamental_26']]

cols_lst = cols_lst[:8]

params_xgb = {'objective'        : 'reg:linear',
              'tree_method'      : 'hist',
              'grow_policy'      : 'depthwise',
              'eta'              : 0.05,
              'subsample'        : 0.4,
              'max_depth'        : 10,
              'min_child_weight' : y_train.size/1000,
              'colsample_bytree' : 1, 
              'base_score'       : 0,
              'silent'           : True,
}
n_round = 34

bst_lst = []
for e, cols in enumerate(cols_lst):
    xgmat_train = xgb.DMatrix(o.train[cols], label=y_train)
    params_xgb['seed'] = 17463 + 543 * e
    bst_lst.append(xgb.train(params_xgb,
                             xgmat_train,
                             num_boost_round=n_round,
                             # __copy__ reduce memory consumption?
                             verbose_eval=False).__copy__())


while True:
    pr_lst = []
    for cols, bst in zip(cols_lst, bst_lst):
        xgmat_test = xgb.DMatrix(o.features[cols])
        pr_lst.append(bst.predict(xgmat_test))

    pred = o.target
    pred['y'] = np.array(pr_lst).mean(0)
    o, reward, done, info = env.step(pred)
    if done:
        print(info)
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
