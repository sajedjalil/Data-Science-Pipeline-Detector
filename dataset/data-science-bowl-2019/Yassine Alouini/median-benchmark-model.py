import pandas as pd 


def median_title_benchmark():
    """ 
    A simple benchmark model that uses the train accuracy_group
    median for each different title. 
    """
    BASE_PATH = "../input/data-science-bowl-2019/"
    TRAIN_PATH = BASE_PATH + "train.csv"
    TRAIN_LABELS_PATH = BASE_PATH + "train_labels.csv"
    TEST_PATH = BASE_PATH + "test.csv"
    train_df = pd.read_csv(TRAIN_PATH)
    train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    # Compute the median by title over the train dataset
    median_title_dict = (train_df.loc[lambda df: (df["type"] == "Assessment")]
                                 .merge(train_labels_df[["installation_id", "game_session", "accuracy_group"]], 
                                       on=["installation_id", "game_session"])
                                 .groupby("title")["accuracy_group"]
                                 .median())
    # Map the train title medians to the test titles and prepare the submission
    # DataFrame.
    submission_df = (test_df.loc[lambda df: (df["type"] == "Assessment")]
                           .sort_values("timestamp")
                           .groupby("installation_id")["title"]
                           .last()
                           .reset_index())
    submission_df["accuracy_group"] = submission_df["title"].map(median_title_dict)
    # Save the submission DataFrame
    submission_df.drop("title", axis=1).to_csv("submission.csv", index=False)
    
    

median_title_benchmark()
    
    
