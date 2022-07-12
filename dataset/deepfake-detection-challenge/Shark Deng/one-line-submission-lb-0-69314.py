import pandas as pd 
import os 
df = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")
df["label"] =  0.5
df.to_csv("submission.csv", index=False)
