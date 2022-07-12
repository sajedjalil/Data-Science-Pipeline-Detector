import pandas as pd


""" 
A python script that transforms the historical transactions CSV dataset into a much smaller one.
For more details, check the accompanying notebook: https://www.kaggle.com/yassinealouini/how-small-could-it-get
"""



# Some constants
PARQUET_ENGINE = "pyarrow"
DATE_COL = "purchase_date"
CATEGORICAL_COLS = ["card_id", "category_3", "merchant_id", "month_lag", 
                    "installments", "state_id", "subsector_id", 
                    "city_id", "merchant_category_id", "merchant_id"]
CATEGORICAL_DTYPES = {col: "category" for col in CATEGORICAL_COLS}
POSITIVE_LABEL = "Y"
INTEGER_WITH_NAN_COL = "category_2"
BINARY_COLS = ["authorized_flag", "category_1"]
INPUT_PATH = "../input/historical_transactions.csv"
OUTPUT_PATH = "historical_transactions.parquet"


def smaller_historical_transactions(input_path, output_path):
    """
    Load the historical transactions CSV file while optimizing the columns' dtypes then save the output 
    into a parquet file. Notice that this function could be made more generic to work with almost any 
    CSV file. Can you see how to do it?
    """
    # Load the CSV file, parse the datetime column and the categorical ones.
    df = pd.read_csv(input_path, parse_dates=[DATE_COL], 
                    dtype=CATEGORICAL_DTYPES)
    # Binarize some columns and cast to the boolean type
    for col in BINARY_COLS:
        df[col] = pd.np.where(df[col] == POSITIVE_LABEL, 1, 0).astype('bool')
    # Cast the category_2 to np.uint8
    df[INTEGER_WITH_NAN_COL] = df[INTEGER_WITH_NAN_COL].values.astype(pd.np.uint8)
    # Save as parquet file
    df.to_parquet(output_path, engine=PARQUET_ENGINE)
    return df
    
def load_historical_transactions(path=None):
    """ 
    Either load the parquet historical transactions file if a path is provided or compute 
    it then load it
    """
    if path is None:
        return smaller_historical_transactions(INPUT_PATH, OUTPUT_PATH)
    else: 
        df = pd.read_parquet(path, engine=PARQUET_ENGINE)
        # Categorical columns aren't preserved when doing pandas.to_parquet
        # (or maybe I am missing something?)
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype('cateogry')
        return df
        
if __name__ == "__main__":
    df = load_historical_transactions()
    print(df.sample(5).T)
    # No more "object" dtypes :)
    assert df.select_dtypes('object').empty
    # The memory footprint in GB should be around 1GB.
    memory_footprint_gb = df.memory_usage(deep="True").sum() / 1024 ** 3
    assert pd.np.round(memory_footprint_gb) == 1.0