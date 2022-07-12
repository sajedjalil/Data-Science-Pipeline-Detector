import numpy as np
import pandas as pd

BASE_PATH = "../input/data-science-bowl-2019/"



def compute_target():
    """ Compute the accuracy group using the train DataFrame. """
    
    TRAIN_PATH = BASE_PATH + "train.csv"

    train_df = pd.read_csv(TRAIN_PATH)

    # In order to compute the accuracy_group, we need first to only keep rows with "Assessment" type. 
    # Afterwards, we filter rows that either have a "title" without "Bird Measurer (Assessment)" and 
    # with event_code 4100 or a "titl" of "Bird Measurer (Assessment)" and event_code 4110.
    # Finally, we make sure that the DataFrame is properly sorted.

    train_df = (train_df.loc[lambda df: df["type"] == "Assessment"]
                        .loc[lambda df: ((df["title"] != "Bird Measurer (Assessment)") & (df["event_code"] == 4100)) | 
                                        ((df["title"] == "Bird Measurer (Assessment)") & (df["event_code"] == 4110))]
                        .sort_values(["installation_id", "timestamp"]))

    # Build the target by counting the times the assessment was solved correctly and incorrectly. 

    train_df["num_correct"] = train_df["event_data"].str.contains('"correct":true').astype(int)
    train_df["num_incorrect"] = train_df["event_data"].str.contains('"correct":false').astype(int)
    
    df = (train_df.groupby(["game_session", "installation_id"])
                  .agg({"num_correct": "sum",
                        "num_incorrect": "sum",
                        "title": "last"})
                  .reset_index())

    # Nested where statements to compute the accuracy group.

    df["accuracy_group"] = np.where((df["num_correct"] == 1) & (df["num_incorrect"] == 0), 3,
                                    np.where((df["num_correct"] == 1) & (df["num_incorrect"] == 1), 2,
                                             np.where((df["num_correct"] == 1) & (df["num_incorrect"] >= 2), 1, 0)))

    assert df["accuracy_group"].isin(range(4)).all()
    
        
    return df

target_df = compute_target()
train_labels_df = pd.read_csv(BASE_PATH + "train_labels.csv")[["game_session", "installation_id", "accuracy_group", "num_correct", "num_incorrect", "title"]]

# Sort

target_df = target_df.sort_values(["game_session", "installation_id"]).reset_index(drop=True).sort_index(axis=1)
train_labels_df = train_labels_df.sort_values(["game_session", "installation_id"]).reset_index(drop=True).sort_index(axis=1)

# Check head

print("Head(2) for computed then provided: ")
print(target_df.head(2).T)
print(train_labels_df.head(2).T)



# Finally, let's print the value counts of title and accuracy_groups

print("Title value counts for computed then provided: ")
print(target_df.title.value_counts().to_dict())
print(train_labels_df.title.value_counts().to_dict())

print("Accuracy group value counts for computed then provided: ")
print(target_df.accuracy_group.value_counts().to_dict())
print(train_labels_df.accuracy_group.value_counts().to_dict())