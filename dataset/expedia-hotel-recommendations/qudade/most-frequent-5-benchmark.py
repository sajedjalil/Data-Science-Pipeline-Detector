import pandas as pd

# find most often occurring hotel_clusters
input = pd.read_csv('../input/train.csv', usecols=['hotel_cluster'])
input['count'] = 1
top5 = (input.groupby(['hotel_cluster'])
              .count()
              .sort_values(by='count', ascending=False)
              .head(5))




# create submission
submission = pd.read_csv('../input/sample_submission.csv')
submission['hotel_cluster'] = ' '.join([str(v) for v in top5.index.values])

submission.to_csv('submission.csv',index=False)