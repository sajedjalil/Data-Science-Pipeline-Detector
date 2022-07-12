import os
import io
import codecs
import pandas as pd
import numpy as np
import string
import operator
from zipfile import ZipFile, is_zipfile

import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager
from string import capwords

# Plotting Options
sns.set_style("whitegrid")
sns.despine()

def plot_bar(df, title, filename):
    p = (
        'Set2', 'Paired', 'colorblind', 'husl',
        'Set1', 'coolwarm', 'RdYlGn', 'spectral'
    )
    bar = df.plot(kind='barh',
                  title=title,
                  fontsize=8,
                  figsize=(12,8),
                  stacked=False,
                  width=1,
                  colors = sns.color_palette(np.random.choice(p), len(df)),
    )

    bar.figure.savefig(filename)
    
    plt.show()

def plot_top_crimes(df, column, title, fname, items=0):
    df.columns     = df.columns.map(operator.methodcaller('lower'))
    by_col         = df.groupby(column) 
    col_freq       = by_col.size()
    col_freq.index = col_freq.index.map(capwords)

    col_freq.sort(ascending=True, inplace=True)
    plot_bar(col_freq[slice(-1, - items, -1)], title, fname)
    
    
def extract_csv(filepath):
    zp  = ZipFile(filepath)
    csv = [f for f in zp.namelist() if os.path.splitext(f)[-1] == '.csv']
    return zp.open(csv.pop())

@contextmanager
def zip_csv_opener(filepath):
    fp = extract_csv(filepath) if is_zipfile(filepath) else open(filepath, 'rb')
    try:
        yield fp
    finally:
        fp.close()

def input_transformer(filepath):
    with zip_csv_opener(filepath) as fp:
        raw = fp.read().decode('utf-8')
    return pd.read_csv(io.StringIO(raw), parse_dates=True, index_col=0, na_values='NONE')

df = input_transformer('../input/train.csv.zip')

print(df.describe())