import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

env = kagglegym.make()
o = env.reset()

excl = ['id', 'sample', 'y', 'timestamp']
cols = [c for c in o.train.columns if c not in excl]

roll_std = o.train.groupby('timestamp').y.mean().rolling(window=10).std().fillna(0)
train_idx = o.train.timestamp.isin(roll_std[roll_std < 0.009].index)

y_train = o.train['y'][train_idx]
xgmat_train = xgb.DMatrix(o.train.loc[train_idx, cols], label=y_train)

# exp070
params_xgb = {'objective'        : 'reg:linear',
              'tree_method'      : 'hist',
              'grow_policy'      : 'depthwise',
              'eta'              : 0.05,
              'subsample'        : 0.6,
              'max_depth'        : 10,
              'min_child_weight' : y_train.size/2000,
              'colsample_bytree' : 1, 
              'base_score'       : y_train.mean(),
              'silent'           : True,
}
n_round = 16

bst_lst = []
for i in range(8):
    params_xgb['seed'] = 2429 + 513 * i
    bst_lst.append(xgb.train(params_xgb,
                             xgmat_train,
                             num_boost_round=n_round,
                             # __copy__ reduce memory consumption?
                             verbose_eval=False).__copy__())

while True:
    pr_lst = []
    xgmat_test = xgb.DMatrix(o.features[cols])
    for bst in bst_lst:
        pr_lst.append(bst.predict(xgmat_test))

    pred = o.target
    pred['y'] = np.array(pr_lst).mean(0)
    o, reward, done, info = env.step(pred)
    if done:
        print(info)
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
