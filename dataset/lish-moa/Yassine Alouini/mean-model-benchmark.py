import pandas as pd
from pathlib import Path

BASE_FOLDER = Path("../input/lish-moa/")
TRAIN_FEATURES_PATH = BASE_FOLDER / "train_features.csv"
TEST_FEATURES_PATH = BASE_FOLDER / "test_features.csv"
TRAIN_TARGETS_PATH = BASE_FOLDER / "train_targets_scored.csv"
SAMPLE_SUBMISSION_PATH = BASE_FOLDER / "sample_submission.csv"



train_targets_df = pd.read_csv(TRAIN_TARGETS_PATH)
train_features_df = pd.read_csv(TRAIN_FEATURES_PATH)
test_features_df = pd.read_csv(TEST_FEATURES_PATH)

sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)


# Since control is always 0, we can filter those

train_sig_ids = train_features_df.loc[lambda df: df["cp_type"] == "ctl_vehicle", "sig_id"].tolist()

mean_train_targets_dict = train_targets_df.loc[lambda df: ~df["sig_id"].isin(train_sig_ids), :].iloc[:, 1:].mean().to_dict()


for col, mean in mean_train_targets_dict.items():
    sample_submission_df.loc[:, col] = mean
    
# For the test, if any are from the control group, we set these to 0
test_sig_ids = test_features_df.loc[lambda df: df["cp_type"] == "ctl_vehicle", "sig_id"].tolist()

sample_submission_df.loc[lambda df: df["sig_id"].isin(test_sig_ids), :].iloc[:, 1:] = 0

sample_submission_df.to_csv("submission.csv", index=False)