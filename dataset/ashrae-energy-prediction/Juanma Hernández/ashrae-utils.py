# !/usr/bin/env python
"""Utility script with functions to be used in ASHRAE
"""

import numpy as np

__author__ = 'Juanma Hernández'
__copyright__ = 'Copyright 2019'
__credits__ = ['Juanma Hernández', 'ArjanGroen', 'ryches']
__license__ = 'GPL'
__maintainer__ = 'Juanma Hernández'
__email__ = 'https://twitter.com/juanmah'
__status__ = 'Utility script'


def reduce_mem_usage(df):
    """
    Based on this great kernel: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    Taken from: https://www.kaggle.com/ryches/simple-lgbm-solution

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    df: DataFrame
    na_list: List
    """
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print('> Memory usage of properties dataframe is: {:.2f} MB\n'.format(start_mem_usg))
    na_list = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            # Print current column type
            print('  column: {}'.format(col))
            print('  dtype before: {}'.format(df[col].dtype))
            # make variables for Int, max and min
            is_int = False
            mn = df[col].min()
            mx = df[col].max()
            print('  min for this col: {}'.format(mn))
            print('  max for this col: {}'.format(mx))
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                na_list.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            as_int = df[col].fillna(0).astype(np.int64)
            result = (df[col] - as_int)
            result = result.sum()
            if -0.01 < result < 0.01:
                is_int = True
                # Make Integer/unsigned Integer datatypes
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            print('  dtype after: ', df[col].dtype, '\n')
    # Print final result
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print('> Memory usage after completion: {:.2f} MB\n'.format(mem_usg))
    print('  This is {:.2%} of the initial size'.format(mem_usg / start_mem_usg))
    return df, na_list
