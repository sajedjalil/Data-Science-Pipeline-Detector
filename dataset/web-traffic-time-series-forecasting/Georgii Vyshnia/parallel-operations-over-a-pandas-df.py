import pandas as pd
import scipy.stats as st
import multiprocessing as mp
import datetime as dt

CHUNKSIZE = 100 # processing 100 rows from the file with training data set at a time

def winsorize_frame(df):
    # process data frame
    for i in range(df.shape[0]):
        page_data = df.iloc[i, 1:-1]
        # Truncate values to the 5th and 95th percentiles via winsorize transformation
        transformed_data = pd.Series(st.mstats.winsorize(page_data, limits=[0.05, 0.05]))
        df.iloc[i, 1:-1] = transformed_data

    function_finish_time = dt.datetime.now()
    print("Fininished a chunk at ", function_finish_time)

    return df

if __name__ == '__main__':
    start_time = dt.datetime.now()
    print("Started at ", start_time)
    print('Reading train data...')
    reader = pd.read_csv("../input/train_1.csv", chunksize=CHUNKSIZE)

    pool = mp.Pool(4) # use 4 processes

    funclist = []
    for df in reader:
        # process each data frame
        f = pool.apply_async(winsorize_frame,[df])
        funclist.append(f)

    result = []
    for f in funclist:
        result.append(f.get(timeout=120)) # timeout in 120 seconds = 2 mins

    # combine chunks with transformed data into a single training set
    training = pd.concat(result)

    end_time = dt.datetime.now()
    elapsed_time = end_time - start_time
    print ("Winsorized training data completely. Elapsed time: ", elapsed_time)