# All credits go to original authors and https://www.kaggle.com/tunguz!
import pandas as pd

test_files = ['../input/lewis-undersampler-9562-version/pred.csv',
              '../input/weighted-app-chanel-os/subnew.csv',
              '../input/single-xgboost-lb-0-965/xgb_sub.csv',
              '../input/lightgbm-with-count-features/sub_lgb_balanced99.csv',
              '../input/deep-learning-support-9663/dl_support.csv',
             '../input/lightgbm-smaller/submission.csv', 
             '../input/do-not-congratulate/sub_mix_logits_ranks.csv',
             '../input/rank-averaging-on-talkingdata/rank_averaged_submission.csv']

model_test_data = []
for test_file in test_files:
    print('read ' + test_file)
    model_test_data.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(model_test_data)

weights = [0.2*0.3*0.07, 0.2*0.3*0.10, 0.2*0.3*0.20, 0.2*0.3*0.65, 0.2*0.3, 0.2*0.4, 0.7,0.1]
print("weights:",weights)
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
print('Done')