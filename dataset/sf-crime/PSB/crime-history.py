import os
import io
import sys
import codecs
import string
import operator

import pandas  as pd
import numpy   as np
import seaborn as sns
import matplotlib.pyplot as plt

from zipfile    import ZipFile, is_zipfile
from contextlib import contextmanager

# Plotting Options
sns.set_style("whitegrid")
sns.despine()

def plot_bar(df, title, filename):
    """
    Helper function for plotting barplots.
    Color selection is made at random from a tuple of seabonrn colorsets
    """
    p = (
        'Set2', 'Paired', 'colorblind', 'husl',
        'Set1', 'coolwarm', 'RdYlGn', 'spectral'
    )
    color = sns.color_palette(np.random.choice(p), len(df))
    bar   = df.plot(kind='barh',
                    title=title,
                    fontsize=6,
                    figsize=(12,8),
                    stacked=False,
                    width=1,
                    color=color,
    )

    bar.figure.savefig(filename)

    plt.show()

def plot_top_crimes(df, column, title, fname, items=0):
    """
    Helper function for plotting seaborn plots
    """
    lower_case     = operator.methodcaller('lower')
    df.columns     = df.columns.map(lower_case)
    by_col         = df.groupby(column)
    col_freq       = by_col.size()
    col_freq.index = col_freq.index.map(string.capwords)

    col_freq.sort(ascending=True, inplace=True)

    plot_bar(col_freq[slice(-1, - items, -1)], title, fname)


def extract_csv(filepath):
    zp  = ZipFile(filepath)
    csv = [
        f for f in zp.namelist()
            if os.path.splitext(f)[-1] == '.csv'
    ]
    return zp.open(csv[0])

@contextmanager
def zip_csv_opener(filepath):
    """
    Context manager for opening zip files.

    Usage
    -----
    with zip_csv_opener(filepath) as fp:
        raw = fp.read()
    """
    fp = extract_csv(filepath) if is_zipfile(filepath) else open(filepath, 'rb')
    try:
        yield fp
    finally:
        fp.close()

def input_transformer(filepath):
    """
    Read file input and transform it into a pandas DataFrame
    """
    with zip_csv_opener(filepath) as fp:
        raw = fp.read()
        raw = raw.decode('utf-8')

    return pd.read_csv(io.StringIO(raw), parse_dates=True, index_col=0, na_values='NONE')

def main(filepath):
    """
    Script Entry Point
    """
    df = input_transformer(filepath)

    plot_top_crimes(df, 'category',   'Top Crime Categories',        'category.png')
    plot_top_crimes(df, 'resolution', 'Top Crime Resolutions',       'resolution.png')
    plot_top_crimes(df, 'pddistrict', 'Police Department Activity',  'police.png')
    plot_top_crimes(df, 'dayofweek',  'Days of the Week',            'weekly.png')
    plot_top_crimes(df, 'address',    'Top Crime Locations',         'location.png', items=20)
    plot_top_crimes(df, 'descript',   'Descriptions',                'descript.png', items=20)


if __name__ == '__main__':
    sys.exit(main('../input/train.csv.zip'))
