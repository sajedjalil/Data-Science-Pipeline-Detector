import pandas as pd
import json


""" A pipeline for processing the data and extracting additional features. 
Can be used for both train and test CSV files. Notice that
the final DataFrame contains string values that should be label and/or one-hot 
encoded (using pandas.factorize and/or pandas.get_dummies respectively).
This will be done in an upcoming script or notebook.
Stay tuned!
"""

# Some constants

# Otherwise, pandas will try to interpret this column as an integer,
# which is wrong according to the competition's guidelines.
VISITOR_ID_COL = "fullVisitorId"
DTYPES = {VISITOR_ID_COL: 'str'}
TARGET_COL = "transactionRevenue"
# Fill the target value NaNs
MISSING_TARGET_VALUE = 0.0
# The timestamp of the trasnaction in Posix format. 
POSIX_COL = "visitStartTime"
TMS_GMT_COL = "tms_gmt"
# The nested columns that will be flattend.
# TODO: Add some comments for the nested columns.
GEO_COL = "geoNetwork"
DEVICE_COL = "device"
TOTALS_COL = "totals"
TRAFFIC_COL = "trafficSource"
# List of columns that are nested that should be flattened in order 
# to be used. 
COLUMNS_TO_FLATTEN = [GEO_COL, DEVICE_COL, TOTALS_COL, TRAFFIC_COL]
# This is needed since there is a nested column wihtin a nested column 
# (the "trafficSource" one)
COLUMNS_TO_FLATTEN_SECOND_LEVEL = ["adwordsClickInfo"]


class DataPipeline(object):
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def _load_data(self):
        """ Load the raw data. Make sure to use the correct types."""
        return pd.read_csv(self.data_path, dtype=DTYPES)

    @staticmethod
    def _flatten_column(df, col):
        # TODO: Add some documentation
        records = []
        for index, row in df[col].iteritems():
            if type(row) == str:
                parsed_row = json.loads(row)
            elif type(row) == dict:
                parsed_row = row
            else:
                parsed_row = row
            records.append(parsed_row)
        return pd.DataFrame(records)

        
    
    def _flatten_columns(self, df, cols):
        # TODO: Refactor the columns unnesting.
        # TODO: Add the other columns to flatten
        dfs = []
        for col in cols:
            print(f"Flattening {col}")
            _df = self._flatten_column(df, col) 
            dfs.append(_df)
        
        return pd.concat(dfs, axis=1)

    def _parse_temporal_column(self, df):
        # TODO: Add some documentation
        return df.assign(**{TMS_GMT_COL: lambda df: pd.to_datetime(df[POSIX_COL], unit='s')})
    
    def _add_calendar_features(self, df):
        # TODO: Update the documentation for this section.
        # TODO: Refactor the extraction ofthe tms_gmt Series for a faster result.
        return (df.assign(hour=lambda df: df[TMS_GMT_COL].dt.hour,
                          doy=lambda df: df[TMS_GMT_COL].dt.day, 
                          dow=lambda df: df[TMS_GMT_COL].dt.dayofweek,
                          woy=lambda df: df[TMS_GMT_COL].dt.week,
                          month=lambda  df: df[TMS_GMT_COL].dt.month,
                          year= lambda df: df[TMS_GMT_COL].dt.year)
                  .drop([TMS_GMT_COL, POSIX_COL, "date"], axis=1))
                  
    @staticmethod
    def _get_constant_columns(df):
        # Inspired from this great kernel: 
        # https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
        # TODO: Add some documentation.
        cols = []
        for col in df.columns:
            try:
                if df[col].nunique(dropna=False)==1:
                    cols.append(col)
            except:
                print(col)
                continue
        return cols

    
    def run(self):

        df = (self._load_data()
                  .pipe(self._parse_temporal_column)
                  .pipe(self._add_calendar_features))
                  
        print("Columns before flattening: ", df.columns.tolist())
        
        flattened_df = self._flatten_columns(df, COLUMNS_TO_FLATTEN)
        df = pd.concat([flattened_df, df], axis=1)
        df = df.drop(COLUMNS_TO_FLATTEN, axis=1)
        
        

        flattened_df = self._flatten_columns(df, COLUMNS_TO_FLATTEN_SECOND_LEVEL)
        df = pd.concat([flattened_df, df], axis=1)
        df = df.drop(COLUMNS_TO_FLATTEN_SECOND_LEVEL, axis=1)
        

        print("Columns after flattening: ", df.columns.tolist())
        
        
        if "train" in self.data_path:
            df[TARGET_COL] = df[TARGET_COL].fillna(MISSING_TARGET_VALUE)
            
        # Drop columns that contain only NaN values.
        df = df.dropna(axis=1, how="all")



        # Get constant columns and drop them
        cst_cols = self._get_constant_columns(df)
        print(f"Constant columns to drop: {cst_cols}")
        df = df.drop(cst_cols, axis=1)
    
        print("DataFrame shape after processing:", df.shape)
        
        return df
        

def compute_and_save_processed_df(input_path):
    """ Run the processing pipeline and save the resulting output.
    Notice that this works for both training and testing CSV files.
    """
    input_type = "train" if "train" in input_path else "test"
    output_data_path = "augmented_{}.csv".format(input_type)
    pipeline = DataPipeline(input_path)
    print("Processing the {} dataset".format(input_type))
    df = pipeline.run()
    assert not df.empty
    print("Saving the processed {} dataset".format(input_type))
    df.to_csv(output_data_path, index=False)
            
        
        
    
    
if __name__ == "__main__":
    PATHS = ["../input/train.csv", "../input/test.csv"]
    for path in PATHS:
        compute_and_save_processed_df(path)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        