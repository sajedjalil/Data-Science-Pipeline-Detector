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

from csv import DictReader
from collections import defaultdict
from datetime import datetime

start_time = datetime.now()

search_destination_hotel_cluster_dict = defaultdict(lambda: defaultdict(int))

for i, row in enumerate(DictReader(open("../input/train.csv"))):
    if row["is_booking"] == "1":
        search_destination_hotel_cluster_dict[row["srch_destination_id"]][row["hotel_cluster"]] += 1
    if i % 1000000 == 0:
        print("%s\t%s"%(i, datetime.now() - start_time))

def get_top_five_as_string(input_array):
    print("%s"%(input_array))
    return " ".join(sorted(input_array, key=input_array.get, reverse=True)[:5])

most_frequent_hotel_clusters = defaultdict(str)

for search_destination_id in search_destination_hotel_cluster_dict:
	most_frequent_hotel_clusters[search_destination_id] = get_top_five_as_string(search_destination_hotel_cluster_dict[search_destination_id])

with open("test_hotel_clusters.csv", "w") as results_file:
	results_file.write("id,hotel_cluster\n")
	for i, row in enumerate(DictReader(open("../input/test.csv"))):
		results_file.write("%d, %s\n"%(i, most_frequent_hotel_clusters[row["srch_destination_id"]]))
		if i % 1000000 == 0:
			print("%s\t%s"%(i, datetime.now() - start_time))



