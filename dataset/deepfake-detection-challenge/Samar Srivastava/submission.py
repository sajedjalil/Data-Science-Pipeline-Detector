import pandas as pd
df = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv',converters={'label':lambda e:.5})
df.to_csv('submission.csv',index=False)