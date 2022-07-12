# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

dtype = {
    'antiguedad':np.object_,
    'fecha_dato':np.object_
}

df = pd.read_csv('../input/train_ver2.csv', usecols=['antiguedad', 'fecha_dato'], dtype=dtype)
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')

print(df.groupby('fecha_dato').antiguedad.mean())
# Any results you write to the current directory are saved as output.