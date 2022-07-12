

# Ensemble 1 - LB : 0.9690
# Ensemble 2 - LB : 0.9694

# Ensemble of the above 2 Ensemble - LB : 0.9696

# All credits go to original authors!

##Ensemble 1 - https://www.kaggle.com/gopisaran/adding-to-the-blender-lb-0-9690
##Ensemble 2 - https://www.kaggle.com/aharless/do-not-congratulate

# I thank Andy Harless for his blend!

# Ensemble of an Ensemble or Can we call this recursive blending/ensembling ?


import pandas as pd

test_files = ['../input/fork-of-adding-to-the-blender-lb-0-9688/average_result.csv',
              '../input/do-not-congratulate/sub_mix.csv']

model_test_data = []
for test_file in test_files:
    print('read ' + test_file)
    model_test_data.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(model_test_data)

weights = [0.5,0.5]
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