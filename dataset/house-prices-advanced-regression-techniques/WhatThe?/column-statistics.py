"""
column_statistics: Integrated reporting of statistics of Pandas DataFrames.

Lists column name, data type, counts of NaN values, counts of distinct
values as well as standard descriptive statistics.

The statistics are reported in a DataFrame of their own, allowing sorting
and selective viewing of data.

It's like a combination of DataFrame.info() and DataFrame.describe()
with skew and kurtosis information added.

Basic Usage:

    stats_df = colstats.stats(Xtrain_df)
    stats_df

Sorting features:

    colstats.sort(stats_df, 'nan_cts').head(6)

Finding features by type:

    colstats.of_type(stats_df, 'object')
"""

import numpy as np
import pandas as pd


def stats(data_frame):
    """
    Collect stats about all columns of a dataframe, their types
    and descriptive statistics, return them in a Pandas DataFrame

    :param data_frame: the Pandas DataFrame to show statistics for
    :return: a new Pandas DataFrame with the statistics data for the
    given DataFrame.
    """
    stats_column_names = ('column', 'dtype', 'nan_cts', 'val_cts',
                          'min', 'max', 'mean', 'stdev', 'skew', 'kurtosis')
    stats_array = []
    for column_name in sorted(data_frame.columns):
        col = data_frame[column_name]
        if is_numeric_column(col):
            stats_array.append(
                [column_name, col.dtype, col.isna().sum(), len(col.value_counts()),
                 col.min(), col.max(), col.mean(), col.std(), col.skew(),
                 col.kurtosis()])
        else:
            stats_array.append(
                [column_name, col.dtype, col.isna().sum(), len(col.value_counts()),
                 0, 0, 0, 0, 0, 0])
    stats_df = pd.DataFrame(data=stats_array, columns=stats_column_names)
    return stats_df


def of_type(stats_data_frame, column_dtype):
    """
    Filter on columns of a given dtype ('object', 'int64', 'float64', etc)

    :param stats_data_frame: a DataFrame produced by the stats() function (above)
    :param column_dtype: a valid column dtype string ('object', 'int64', 'float64', ...)
    :return: the stats_data_frame that was passed in
    """
    return stats_data_frame[stats_data_frame['dtype'] == column_dtype]


def sort(data_frame, column_name, ascending=False):
    """
    Shorthand for sorting a data frame by one column's values.
    Useful with the status dataframe columns.

    :param data_frame: data_frame whose contents are to be sorted
    :param column_name: String name of the column to sort by
    :param ascending: if True, sort in ascending order (default, False)
    :return: a copy of the data_frame, sorted as specified
    """
    return data_frame.sort_values(column_name, ascending=ascending)


def is_numeric_column(df_column):
    """
    Answer whether a column of a data_frame is numeric

    :param df_column: Any column from a Pandas DataFrame
    :return: True if it's in one of the standard numeric types
    """
    numeric_types = (np.int16, np.float16, np.int32, np.float32,
                     np.int64, np.float64)
    return df_column.dtype in numeric_types
