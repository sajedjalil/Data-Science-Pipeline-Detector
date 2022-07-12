# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

try:
    import sys

    sys.path.append("../input/blindnessdetection15/blindness-detection-master (1)/blindness-detection-master")

    from submit import main

    main("../input/baeff9f285e140868c360f60b5e12092/2-12-checkpoint.pth", samples_to_visualise=0)
except Exception as e:
    print(e)
    