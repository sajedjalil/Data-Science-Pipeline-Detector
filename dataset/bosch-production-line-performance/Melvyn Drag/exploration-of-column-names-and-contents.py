"""
This script explores some basic questions about the columns to help decide how to make a classifier.
"""

numeric = "../input/train_numeric.csv"
categorical = "../input/train_categorical.csv"
# omit date, because date corresponds to the previous two.

question_zero = "Is the same feature measured multiple times? {}"

question_one = "Is the same feature measured at multiple lines? {}"

question_two = "Is the same feature measured at multiple stations? {}"

question_three = "Are certain flow paths more prone to errors than others? {}" # probably. going to use the flow path code from John to see if thats the case.

with open(numeric, "r") as fin:
    numeric_cols = fin.readline()
    
with open(categorical, "r") as fin:
    categorical_cols = fin.readline()

numeric_cols = numeric_cols.split(",")
categorical_cols = categorical_cols.split(",")

numeric_cols = [nc for nc in numeric_cols if len(nc.split("_")) == 3]
categorical_cols = [cc for cc in categorical_cols if len(cc.split("_")) == 3]

numeric_col_set = set(numeric_cols)
categorical_col_set = set(categorical_cols)
assert(len(numeric_col_set.intersection(categorical_col_set)) == 0) # Sanity check. Make sure they don't have identical columns.

def make_ints(list_of_tups):
    for idx, tup in enumerate(list_of_tups):
        empty = []
        for element in tup:
            empty.append(int(element[1:]))
        list_of_tups[idx] = tuple(empty)
    return list_of_tups

print("There are {} numeric columns".format(len(numeric_cols)))
print("There are {} categorical columns".format(len(categorical_cols)))
all_cols = numeric_cols + categorical_cols
l_s_f_tups = [tuple(c.split("_")) for c in all_cols]
l_s_f_tups = make_ints(l_s_f_tups)
l_s_f_tups = sorted(l_s_f_tups, key = lambda t: t[2])

from itertools import groupby
features_measured_more_than_once = []
for key, group in groupby(l_s_f_tups, lambda x: x[2]):
    if len(list(group)) > 1:
        features_measured_more_than_once.append(group)


# Question 1
if len(features_measured_more_than_once) > 0:
    print(question_zero.format("Yes"))
else:
    print(question_zero.format("No"))
    print(question_one.format("No"))
    print(question_two.format("No"))
    




    