# All credits go to original authors!

import pandas as pd
import numpy as np

print('2 momths are left to end this competition and blends are giving better results than single models.')
print('looks like this competition will also end up like toxic comment classifier challenge. Blends, blends everywhere !!')

test_files = ['../input/lewis-undersampler-9562-version/pred.csv',
              '../input/weighted-app-chanel-os/subnew.csv',
              '../input/single-xgboost-lb-0-964/xgb_sub.csv',
              '../input/lightgbm-fixing-unbalanced-data-auc-0-9787/sub_lgb_balanced99.csv',
              '../input/lightgbm-with-count-features/sub_lgb_balanced99.csv',
              '../input/deep-learning-support/dl_support.csv',
              '../input/lightgbm-smaller/submission.csv']

ll = []
for test_file in test_files:
    print('read ' + test_file)
    ll.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(ll)

weights = [0.3*0.09, 0.3*0.14, 0.3*0.18, 0.3*0.09, 0.3*0.5, 0.3, 0.4]
cc = 'is_attributed'
print(np.corrcoef([ll[0][cc], ll[3][cc], ll[4][cc], ll[5][cc], ll[6][cc]]))
print('ALWAYS BLEND NON CORRELATED RESULTS TO PREVENT OVERFITTING..')

print('predict')
test_predict_column = [0.] * len(ll[0][cc])
for ind in range(0, n_models):
    test_predict_column += ll[ind][cc] * weights[ind]

print('make result')
final_result = ll[0]['click_id']
final_result = pd.concat((final_result, pd.DataFrame(
    {cc: test_predict_column})), axis=1)
final_result.to_csv("blend.csv", index=False)
