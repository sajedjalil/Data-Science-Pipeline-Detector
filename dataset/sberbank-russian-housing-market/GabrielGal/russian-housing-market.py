# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
import matplotlib.pyplot as plt
 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

training_set = pd.read_csv("../input/train.csv")
testing_set = pd.read_csv("../input/test.csv")

training_features = training_set.drop("price_doc", axis = 1)
training_outputs = training_set["price_doc"]

# Preprocessing and cleaning the data

# Columns that contain data with NA values

contains_NA_features = ["life_sq",
                        "floor", 
                        "max_floor", 
                        "material", 
                        "build_year", 
                        "num_room", 
                        "kitch_sq", 
                        "state",
                        "preschool_quota",
                        "school_quota",
                        "hospital_beds_raion",
                        "raion_build_count_with_material_info",
                        "build_count_block",
                        "build_count_wood",
                        "build_count_frame",
                        "build_count_brick",
                        "build_count_monolith",
                        "build_count_panel",
                        "build_count_foam",
                        "build_count_slag",
                        "build_count_mix",
                        "raion_build_count_with_builddate_info",
                        "build_count_before_1920",
                        "build_count_1921-1945",
                        "build_count_1946-1970",
                        "build_count_1971-1995",
                        "build_count_after_1995",
                        "metro_min_walk",
                        "metro_km_walk",
                        "railroad_station_walk_km",
                        "railroad_station_walk_min",
                        "ID_railroad_station_walk",
                        "cafe_sum_500_min_price_avg",
                        "cafe_sum_500_max_price_avg",
                        "cafe_avg_price_500",
                        "cafe_sum_1000_min_price_avg",
                        "cafe_sum_1000_max_price_avg",
                        "cafe_avg_price_1000",
                        "cafe_sum_1500_min_price_avg",
                        "cafe_sum_1500_max_price_avg",
                        "cafe_avg_price_1500",
                        "cafe_sum_2000_min_price_avg",
                        "cafe_sum_2000_max_price_avg",
                        "cafe_avg_price_2000",
                        "cafe_sum_3000_min_price_avg",
                        "cafe_sum_3000_max_price_avg",
                        "cafe_avg_price_3000",
                        "prom_part_5000",
                        "cafe_sum_5000_min_price_avg",
                        "cafe_sum_5000_max_price_avg",
                        "cafe_avg_price_5000"]

# Adding a binary column that will contain 0 or 1 depending on whether the value in referenced column has a NA

# Determine which areas have outliers or incorrect data

contains_outliers = training_features.select_dtypes(exclude = ["object"]).keys()

contains_outliers = np.log(training_features[contains_outliers])

for column in contains_outliers:
    contains_outliers[column] = [0.0 if np.isinf(x) == True else x for x in contains_outliers[column]]

for column in contains_outliers:
    Q1 =np.percentile(contains_outliers[column], 25)
    Q3 =np.percentile(contains_outliers[column], 75)
    step = 3 * (Q3 - Q1)
    print ("Data points considered outliers for the feature '{}':".format(column))
    display(contains_outliers[~((contains_outliers[column] >= Q1 - step) & (contains_outliers[column] <= Q3 + step))])

'''
for column in contains_NA_features:
    training_features.insert(training_features.columns.get_loc(column), column + "_NA_Decision", training_features[column])
    training_features[column + "_NA_Decision"] = [0 if np.isnan(x) == True else 1 for x in training_features[column + "_NA_Decision"]]
'''
# scaler = MinMaxScaler()

# training_features[normalize_features] = scaler.fit_transform(training_features[normalize_features])

# Any results you write to the current directory are saved as output.