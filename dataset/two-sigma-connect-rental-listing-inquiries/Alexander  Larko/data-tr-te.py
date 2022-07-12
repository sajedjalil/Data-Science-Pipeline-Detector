
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
def read_train_test():
    data_path = "../input/"
    train_file = data_path + "train.json"
    test_file = data_path + "test.json"
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)
    return train_df, test_df

train_df, test_df = read_train_test()
import math
def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho


def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi


def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)


def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)


def add_rotation(degrees, df):
    namex = "rot" + str(degrees) + "_X"
    namey = "rot" + str(degrees) + "_Y"

    df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

    return df

def operate_on_coordinates(tr_df, te_df):
    for df in [tr_df, te_df]:
        #polar coordinates system
        df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        #rotations
        #for angle in [15,30,45,60]:
        #    df = add_rotation(angle, df)

    return tr_df, te_df

train_df, test_df = operate_on_coordinates(train_df, test_df)

import re

def cap_share(x):
    return sum(1 for c in x if c.isupper())/float(len(x)+1)

for df in [train_df, test_df]:
    # do you think that users might feel annoyed BY A DESCRIPTION THAT IS SHOUTING AT THEM?
    df['num_cap_share'] = df['description'].apply(cap_share)
    
    # how long in lines the desc is?
    df['num_nr_of_lines'] = df['description'].apply(lambda x: x.count('<br /><br />'))
   
    # is the description redacted by the website?        
    df['num_redacted'] = 0
    df['num_redacted'].ix[df['description'].str.contains('website_redacted')] = 1

    
    # can we contact someone via e-mail to ask for the details?
    df['num_email'] = 0
    df['num_email'].ix[df['description'].str.contains('@')] = 1
    
    #and... can we call them?
    
    reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
    def try_and_find_nr(description):
        if reg.match(description) is None:
            return 0
        return 1

    df['num_phone_nr'] = df['description'].apply(try_and_find_nr)



feat_to_use = ["num_phi","num_rho",'num_cap_share','num_nr_of_lines','num_redacted','num_email','num_phone_nr',"listing_id"]

train_df = train_df[feat_to_use]
test_df = test_df[feat_to_use]
train_df.to_csv('train_last_feat.csv', index=False)
test_df.to_csv('test_last_feat.csv', index=False)
