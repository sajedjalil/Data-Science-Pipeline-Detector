# All credits go to original authors!

## Old Kerlnel for 4th CSV  - https://www.kaggle.com/gopisaran/better-train-sample-only-18-m-records-lb-0-9680
## New Kerlnel for 4th CSV -  https://www.kaggle.com/gopisaran/one-more-feature-lb-0-9682


import pandas as pd

test_files = ['../input/lewis-undersampler-9562-version/pred.csv',
              '../input/weighted-app-chanel-os/subnew.csv',
              '../input/single-xgboost-lb-0-965/xgb_sub.csv',
              '../input/one-more-feature-lb-0-9682/sub_lgb_balanced99.csv',
              '../input/lightgbm-smaller/submission.csv']

model_test_data = []
for test_file in test_files:
    print('read ' + test_file)
    model_test_data.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(model_test_data)

weights = [0.3*0.05, 0.3*0.10, 0.3*0.20, 0.495, 0.4]
column_name = 'is_attributed'

print('predict')
test_predict_column = [0.] * len(model_test_data[0][column_name])
for ind in range(0, n_models):
    test_predict_column += model_test_data[ind][column_name] * weights[ind]

print('make result')
final_result = model_test_data[0]['click_id']
final_result = pd.concat((final_result, pd.DataFrame(
    {column_name: test_predict_column})), axis=1)
final_result.to_csv("average_result.csv", index=False)
