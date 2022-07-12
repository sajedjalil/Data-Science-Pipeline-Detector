import pandas as pd

DATA_FOLDER = "../input/gapdatasets/" # Folder that contains the 3 datasets from the Github Repo (https://github.com/xaalai/gap-coreference)
CLEANED_DATA_FOLDER = "../kaggle/working/data_with_none/" # Output directory, should already be created before executing the script

def are_AB_both_true(df):
    """ Checks whether A-coref and B-coref can be both true
        In that case there would probably be a problem with our None column rule construction
    """
    return (df["A-coref"] == df["B-coref"] == True).sum()

def add_none_column(df):
    """ Creates a None column in the dataframe if A-coref and B-coref are both False
        We already know that they can not be both True
    """
    df["None"] = df['A-coref'] == df['B-coref']
    return df

if __name__ == "__main__":
    filenames = [
        "gap-development.tsv", 
        "gap-test.tsv", 
        "gap-validation.tsv"
    ]
    for filename in filenames:
        file = DATA_FOLDER + filename
        df = pd.read_csv(file, sep="\t")
        df_with_none = add_none_column(df)
        df_with_none.to_csv(f"{filename}", sep="\t")