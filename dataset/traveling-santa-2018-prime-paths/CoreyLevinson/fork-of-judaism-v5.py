import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

debug = False
NUMBER_OF_PRIMES = 17802

nrows = None
if debug:
    nrows = 10000

df = pd.read_csv('../input/cities.csv', nrows=nrows)

# This implementation was copied from: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
df = reduce_mem_usage(df)



ixy  = df.values # CityId(or index), X, Y
R    = []        # Result list of CityId

del df
gc.collect()

# Prime function from https://stackoverflow.com/questions/18833759/python-prime-number-checker
import math
def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    if n < 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

for i in tqdm(range(len(ixy))):
    d    = (ixy[:, 1] - ixy[0, 1]) ** 2 + (ixy[:, 2] - ixy[0, 2]) ** 2 # the distance from last choiced city
    ixyd = np.concatenate([ixy, d.reshape(-1, 1)], axis=1)
    
    del d
    gc.collect()
    
    argi = np.argsort(ixyd[:, 3]) # Argsorted index by the distance
    ixyd = ixyd[argi]
    
    # If it's time for a prime number
    if i % 10 == 0 and NUMBER_OF_PRIMES > 0 and i!=0:
        prime_number = -1
        j_used = -1
        for j in range(len(ixy)):
            if is_prime(int(ixyd[j,0])):
                prime_number = int(ixyd[j,0])
                j_used = j
                break
        R.append(prime_number)
        ixy = np.concatenate([ixyd[0:j_used, :-1], ixyd[(j_used+1):, :-1]])
        NUMBER_OF_PRIMES = NUMBER_OF_PRIMES - 1 # Subtract total number of primes
        
    else:
        nonprime_number = -1
        j_used = -1
        for j in range(len(ixy)):
            if not is_prime(int(ixyd[j,0])):
                nonprime_number = int(ixyd[j,0])
                j_used = j
                break
        R.append(nonprime_number)
        ixy = np.concatenate([ixyd[0:j_used, :-1], ixyd[(j_used+1):, :-1]])
        
if not debug:
    s = pd.read_csv('../input/sample_submission.csv')
    s['Path'] = np.array(R + [0]) # Return 0
    s.to_csv('simple_nearest.csv', index=False)
else:
    s = pd.read_csv('../input/sample_submission.csv', nrows=(nrows+1))
    s['Path'] = np.array(R + [0]) # Return 0
    s.to_csv('simple_nearest.csv', index=False)
