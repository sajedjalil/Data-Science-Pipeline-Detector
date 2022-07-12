# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


##borrowing major ideas from Metabaron's script
prod_tab = pd.read_csv('../input/producto_tabla.csv')
prod_tab['weight'] = prod_tab['NombreProducto'].str.extract(r'(\d+\s?(kg|Kg|g|G))')[0]
prod_tab['volume'] = prod_tab['NombreProducto'].str.extract(r'(\s?\d+\s?(ml)\s)')[0]
prod_tab['pieces'] = prod_tab['NombreProducto'].str.extract(r'(\s?\d+(p|P))')[0]
prod_split = prod_tab.NombreProducto.str.split(r"(\s\d+\s?(kg|Kg|g|G|in|ml|pct|p|P|Reb))")
prod_tab['product'] = prod_split.apply(lambda x: x[0])
prod_tab['brand'] = prod_split.apply(lambda x: x[-1]).str.split().apply(lambda x: x[:-1])
prod_tab['num_brands'] = prod_tab.brand.apply(lambda x: len(x))
prod_tab['prod_split'] = prod_tab['product'].str.split(r'[A-Z][A-Z]').apply(lambda x: x[0])


prod_tab.to_csv('prod_tab.csv')