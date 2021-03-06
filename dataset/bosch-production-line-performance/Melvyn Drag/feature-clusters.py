import sys

def get_flow_path(iterable):
    return tuple([idx for idx, element in enumerate(iterable) if element != ""])

numeric="../input/train_numeric.csv"
categorical="../input/train_categorical.csv"
#
#
# Question 1: Are there any meaningful clusters by the measured features?
# An interesting extension to this is to see if the features are unique to the line and station they are measured at...
# ... this is left by the author as an exercise to the reader.
#
#

flow_paths = {}
with open(numeric, "r") as n:
    with open(categorical, "r") as c:
        n.readline()
        c.readline()
        counter = 0
        for n_row, c_row in zip(n, c):
            counter += 1
            if counter % 1000 == 0:
                print("processing_row: {}".format(counter))
            n_row = n_row.strip()
            c_row = c_row.strip()
            n_id, n_row = n_row.split(",")[0], n_row.split(",")[1:]
            c_id, c_row = c_row.split(",")[0], c_row.split(",")[1:]
            assert(n_id == c_id)
            idx = n_id
            whole_row = n_row + c_row
            fp = get_flow_path(whole_row)
            if fp in flow_paths.keys():
                flow_paths[fp].append(idx)
            else:
                flow_paths[fp] = [idx]

#
#
# Question 2: How big are the clusters generated by this grouping method?
# What is the error rate per cluster...
# ... This potentially interesting result is also left as an exercise to the reader ...
#
#

import json
with open("flow_path.json", "w") as fout:
    for key, value in flow_paths.items():
        fout.write(str(key) + "," + str(value)+"\n")


import matplotlib.pyplot as plt
flow_path_length = [len(value) for key, value in flow_paths.items()]
flow_path_length.sort()
x_values = list(range(len(flow_path_length)))
plt.plot(x_values, flow_path_length)
plt.title("Show the cluster sizes")
plt.savefig("cluster_sizes.png")
