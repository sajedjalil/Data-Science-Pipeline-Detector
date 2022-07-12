# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def readNumericCsvRandom(fname, total_lines, sample_size=500000, header_row=True,
                         replace=True, **kwargs):
    '''
    Parses a csv, taking a random sample of rows, with uniform distribution.
    @param fname: csv filename.
    @param total_lines: Total lines in file.
    @param sample_size: Sample size.
    @param header_row: Wether the first row corresponds to a header.
    @param replace: Speed up sampling by allowing repeated entries.
    If `total_lines` is comparatively bigger than `sample_size`, you will
    obtain approximately the same sample size.
    @param **kwargs: Keyword arguments passed to `DataFrame` constructor.
    @returns: pandas dataframe. Original row numbers are injected in the
    returned DataFrame.
    '''
    sample = np.random.choice(total_lines, sample_size, replace=replace)
    if replace:
      sample = np.unique(sample)
    sample.sort()
    data = StringIO()
    with open(fname) as f:
        if header_row:
            header = 'id,' + f.readline()
            data.write(header)
        cumsum = 0
        for val in sample:
            skip = val - cumsum
            for _ in range(skip - 1):
                f.readline()
            _data = '{0},'.format(val) + f.readline()
            data.write(_data)
            cumsum = val
        data.seek(0)
        return pd.read_csv(data,**kwargs)
        

