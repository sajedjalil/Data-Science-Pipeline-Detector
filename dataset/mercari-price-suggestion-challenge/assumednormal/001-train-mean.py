import pandas as pd

train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df = pd.read_csv("../input/test.tsv", sep="\t")

submission_df = test_df[["test_id"]].copy()
submission_df["price"] = train_df["price"].mean()

submission_df.to_csv("001_train_mean_submission.csv", index=False)