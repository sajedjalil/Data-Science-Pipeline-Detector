import pandas as pd

train_df = pd.read_csv('../input/act_train.csv')
test_df = pd.read_csv('../input/act_test.csv')
result_df = (train_df.groupby(['activity_category']))['outcome'].mean().reset_index()
result_df = pd.merge(test_df,result_df,on='activity_category',how='left')
result_df[['activity_id','outcome']].to_csv('beatbenchmark.csv',index=False)
