import numpy as np
import gc
from os.path import basename
import pyarrow.parquet as pq
from numba import jit
from joblib import delayed, Parallel
from tqdm import tqdm

# This script is thought of as an initial feature generating script you would run once

# Features and inspiration come from 
# https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694

CHUNK_LEN = 5000
N_JOBS = -1
COL_STEP = 45  # must be divisible by three


@jit
def calc_stats(_arr):
    # add whatever you like
    mean = _arr.mean(axis=2)
    std = _arr.std(axis=2)
    std_top = std + mean
    std_bottom = std - mean
    return mean, std, std_top, std_bottom


def featurize(filename, col_range):
    start_col, n_cols = col_range
    x_list = []
    with Parallel(n_jobs=N_JOBS, max_nbytes='100M', temp_folder='tmp') as prll:
        for i in tqdm(range(start_col, n_cols, COL_STEP)):
            train_df = \
                pq.read_pandas(filename,
                               columns=[str(c) for c in range(i, min(i + COL_STEP, n_cols))]
                               ).to_pandas()

            # get data in shape
            arr = train_df.values.reshape(int(8e5), -1, 3).transpose([1, 0, 2])
            # standardize
            arr = arr.astype(np.float32) / 128
            # choose chunk size (i.e. subsequence length)
            n_chunks = int(np.floor(arr.shape[1] / CHUNK_LEN))

            # aggregate stats
            # n_samples x n_chunks x chunk_len x n_features 
            arr = arr.reshape(arr.shape[0], n_chunks, CHUNK_LEN, arr.shape[2])

            stats_tup = calc_stats(arr)
            
            percentiles_list = [0, 1, 25, 50, 75, 99, 100]
            # create numpy placeholder such that np.percentile won't create float64 arrays
            percentiles = [np.zeros((arr.shape[0], arr.shape[1], arr.shape[3]),
                                    dtype=np.float32) for _ in percentiles_list]

            prll(delayed(np.percentile)(arr, p, axis=2, out=percentiles[i])
                  for i, p in enumerate(percentiles_list))
            del arr
            
            ## add more features here, that aren't "jittable"
            
            mean = stats_tup[0]
            percentiles_demean = [p - mean for p in percentiles]
            x_list.append(np.dstack([a for a in stats_tup] +
                                     percentiles + percentiles_demean))

        x_arr = np.vstack(x_list)
        x_arr.dump(f'{CHUNK_LEN}er_chunks_'
                   f'{basename(filename).split(".")[0]}.pkl')
        del x_arr, percentiles, percentiles_demean, x_list
        gc.collect()


def main():
    print('featurize train set')
    train_col_range = (0, 8712)  # end exclusive
    featurize('../input/train.parquet', train_col_range)

    print('featurize test set')
    test_col_range = (8712, 29049)  # end exclusive
    featurize('../input/test.parquet', test_col_range)


if __name__ == '__main__':
    main()
