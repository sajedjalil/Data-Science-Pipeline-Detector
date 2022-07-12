#Data 
#Generated with scikit learn, 100 Extremely randomized trees as classifier.
#The result is the class probability (of being 1, not 0).  

#Hyper Parameters:
#criterion='entropy', max_depth=23, min_samples_leaf=4, max_features='sqrt',

#Might help with your Ensembles or as a comparison to other decision tree methods.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
output = pd.read_csv('../input/extremely-randomized-trees-classification/kernel.csv')
output.to_csv('./submission.csv')