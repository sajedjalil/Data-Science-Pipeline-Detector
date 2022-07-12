import pandas as pd


test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
gt_df = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv")


gt_df = gt_df[['choose_one', 'text']]
gt_df['target'] = (gt_df['choose_one']=='Relevant').astype(int)
gt_df['id'] = gt_df.index
gt_df

merged_df = pd.merge(test_df, gt_df, on='id')
merged_df

subm_df = merged_df[['id', 'target']]
subm_df

subm_df.to_csv('submission.csv', index=False)