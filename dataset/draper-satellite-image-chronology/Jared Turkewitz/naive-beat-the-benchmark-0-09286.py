import pandas as pd
sub = pd.read_csv('../input/sample_submission.csv')
sub['day'] = "1 5 3 2 4"
sub.to_csv('naive_submit.csv', index=False)