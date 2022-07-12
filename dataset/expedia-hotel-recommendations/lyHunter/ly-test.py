# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

#train = pd.read_csv("../input/train.csv", dtype={   'id':np.int64,
#                                                    'date_time':object,
#                                                    'site_name':np.uint8,
#                                                    'posa_continent':np.uint8,
#                                                    'user_location_country':np.uint8,
#                                                    'user_location_region':np.int16,
#                                                    'user_location_city':np.int32,
#                                                    'orig_destination_distance':np.float32,
#                                                    'user_id':np.int32,
#                                                    'is_mobile':bool,
#                                                    'is_package':bool,
#                                                    'channel':np.int8,
#                                                    'srch_ci':np.object,
#                                                    'srch_co':np.object,
#                                                    'srch_adults_cnt':np.uint8,
#                                                    'srch_children_cnt':np.uint8,
#                                                    'srch_rm_cnt':np.uint8,
#                                                    'srch_destination_id':np.uint16,
#                                                    'srch_destination_type_id':np.int8,
#                                                    'hotel_continent':np.int16,
#                                                    'hotel_country':np.uint8,
#                                                    'hotel_market':np.int16,
#                                                    'is_booking':bool,
#                                                    'cnt':np.int16,
#                                                    'hotel_cluster':np.int16},
#                                                     nrows=2)

a = np.array([[1, 1], [2, 2], [3, 3]])
print(a)
np.insert(a, 1, 5)
print(a)


