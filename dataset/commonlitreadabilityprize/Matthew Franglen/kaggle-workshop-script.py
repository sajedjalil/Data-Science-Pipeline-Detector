# %% [code] {"jupyter":{"outputs_hidden":false}}
import pandas as pd

df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/test.csv")
df = df[["id"]].copy()
df["target"] = 0.0

df.to_csv("/kaggle/working/submission.csv")