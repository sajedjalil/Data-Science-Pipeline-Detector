import pandas as pd
val_df = pd.read_pickle('../input/training-and-validation-data-pickle/validation.pkl.gz')
val_df.to_csv('validation.csv', index=False)