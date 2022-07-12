import pandas as pd

# Load training data
df = pd.read_csv('../input/train.csv', index_col=0)
print(df)