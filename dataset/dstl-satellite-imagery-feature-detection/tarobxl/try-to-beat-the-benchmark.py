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

"""
 1   Buildings - large building, residential, non-residential, fuel storage facility, fortified building
 2   Misc. Manmade structures 
 3   Road 
 4   Track - poor/dirt/cart track, footpath/trail
 5   Trees - woodland, hedgerows, groups of trees, standalone trees
 6   Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
 7   Waterway 
 8   Standing water
 9   Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
 10   Vehicle Small - small vehicle (car, van), motorbike
"""
sub = pd.read_csv('../input/sample_submission.csv')

#POL = "POLYGON ((0 0, 0.009188 0, 0.009188 -0.009039999999999999, 0 -0.009039999999999999, 0 0))"
#sub['MultipolygonWKT'] = POL

POL = "MULTIPOLYGON EMPTY"
sub.loc[sub['ClassType'] == 1,'MultipolygonWKT'] = POL
sub.loc[sub['ClassType'] == 2,'MultipolygonWKT'] = POL
sub.loc[sub['ClassType'] == 3,'MultipolygonWKT'] = POL

sub.loc[sub['ClassType'] == 7,'MultipolygonWKT'] = POL

sub.loc[sub['ClassType'] == 9,'MultipolygonWKT'] = POL
sub.loc[sub['ClassType'] == 10,'MultipolygonWKT'] = POL

sub.to_csv('beat_benchmark.csv', index=False)