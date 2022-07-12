import pandas as pd
import numpy as np
import time



# From here (remastered): https://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, (time2 - time1)))

        return ret
    
    return wrap

BASE_PATH = "../input/data-science-bowl-2019"


def process_df(input_df):

    # In order to compute the accuracy_group, we need to filter title without "Bird Measurer (Assessment)"
    # and with event_code 4100 and "Bird Measurer (Assessment)" with event_code 4110.
    # Then, we make sure that the DataFrame is properly sorted.

    input_df = (input_df.loc[lambda df: df["type"] == "Assessment"]
                        .loc[lambda df: ((df["title"] != "Bird Measurer (Assessment)") & (df["event_code"] == 4100)) |
                                        ((df["title"] == "Bird Measurer (Assessment)") & (df["event_code"] == 4110))]
                        .sort_values(["installation_id", "timestamp"]))

    # Build the target by counting the times the assessment was solved correctly. 

    input_df["num_correct"] = input_df["event_data"].str.contains('"correct":true').astype(int)
    input_df["num_incorrect"] = input_df["event_data"].str.contains('"correct":false').astype(int)
    df = (input_df.groupby(["game_session", "installation_id"])
                  .agg({"num_correct": "sum",
                        "num_incorrect": "sum",
                        "title": "last",
                        "event_id": "count",
                        "timestamp": "last"})
                  .rename(columns={"event_id": "num_games"})
                  .reset_index())

    # Nested where to compute the accuracy group.
    df["accuracy_group"] = np.where((df["num_correct"] == 1) & (df["num_incorrect"] == 0), 3,
                                    np.where((df["num_correct"] == 1) & (df["num_incorrect"] == 1), 2,
                                             np.where((df["num_correct"] == 1) & (df["num_incorrect"] >= 2), 1, 0)))

    assert df["accuracy_group"].isin(range(4)).all()

    return df 

@timing
def features_engineering(processed_df):

    dfs = []
    for index, g in processed_df.groupby(["installation_id", "title"]):

        _df = (g.set_index("timestamp")
               .shift()
               .expanding()
               .agg({"num_games": "sum", "accuracy_group": "mean", "num_correct": "sum", "num_incorrect": "sum"})
               .rename(columns={"num_games": "total_past_games_per_title",
                                "accuracy_group": "mean_past_accuracy_group_per_title",
                                "num_correct": "total_past_num_correct_per_title",
                                "num_incorrect": "total_past_num_incorrect_per_title"})
               .reset_index())

        dfs.append(_df)
        _df["installation_id"] = index[0]
        _df["title"] = index[1]

    title_df = pd.concat(dfs)

    dfs = []
    for index, g in processed_df.groupby(["installation_id"]):

        _df = (g.set_index("timestamp")
               .shift()
               .expanding()
               .agg({"num_games": "sum", "accuracy_group": "mean", "num_correct": "sum", "num_incorrect": "sum"})
               .rename(columns={"num_games": "total_past_games",
                                "accuracy_group": "mean_past_accuracy_group",
                                "num_correct": "total_past_num_correct",
                                "num_incorrect": "total_past_num_incorrect"})
               .reset_index())

        dfs.append(_df)
        _df["installation_id"] = index

    df = pd.concat(dfs)

    return (processed_df.merge(title_df, on=["installation_id", "title", "timestamp"], how="left")
                        .merge(df, on=["installation_id", "timestamp"], how="left"))


if __name__ == "__main__":
    train_features_df = pd.read_csv(BASE_PATH + "/train.csv").pipe(process_df).pipe(features_engineering)
    test_features_df = pd.read_csv(BASE_PATH + "/test.csv").pipe(process_df).pipe(features_engineering)
    print(train_features_df.sample(2).T)
    print(test_features_df.sample(2).T)
    print(train_features_df.corr())
    print(test_features_df.corr())

