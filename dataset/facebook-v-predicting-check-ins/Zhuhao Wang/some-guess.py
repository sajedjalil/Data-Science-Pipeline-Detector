# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial.ckdtree import cKDTree as KDTree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv").as_matrix()
train_tree = KDTree(train_data[:, 1:3])
test_data = pd.read_csv("../input/test.csv").as_matrix()

_, indss = train_tree.query(test_data[:, 1:3], 3)
with open("output.csv", "w") as output:
    output.write("row_id,predictions\n")
    for i, inds in enumerate(indss):
        result = "%d," % int(test_data[i, 0])
        result += " ".join([str(int(ind)) for ind in inds])
        result += "\n"
        output.write(result)