import pandas as pd
import numpy as np

from itertools import tee, izip
import os

# a helper function takes an iterable, return its stepwise pair tuple in a list
# [1, 2, 3, 4, 5] -> [(1, 2), (2, 3), (3, 4), (4, 5)]
def pairwise(iterable):
    floor, ceiling = tee(iterable)
    next(ceiling, None)
    return izip(floor, ceiling)

# a helper function takes a df, a column name, a list of floor/ceiling values 
# "split" the df based on the given column, using the list of floor/ceiling values, return list of df
#
# flr_clg will be floor EXclusive, ceiling INclusive - need special treatment for the first split
def split_df_rows_on_col_ranges(df, col, flr_clg):
    splitted_df = []
    first = True
    for fc in flr_clg:
        if first:
            splitted_df.append(df[ (df[col] >= fc[0]) & (df[col] <= fc[1]) ])
            first = False
        else:
            splitted_df.append(df[ (df[col] > fc[0]) & (df[col] <= fc[1]) ])
    return splitted_df
    
'''
input parameters for def get_grids():
- filename - input filename with path (probably the training set)
- outputPath - only used if the 3rd parameter is set to True, will be the path to store NxN files, 
               each file contains a grid of data points
               
- outputFile - boolean that tells whether you want NxN files as output or a dict of pd.DataFrame
               as output, format would be (x_idx, y_idx) : df_for_grid. If you want file as output
               then x_idx, y_idx will appear in output files' name
               
- n - NxM grid, the N value, for x axis
- m - NxM grid, the M value, for y axis
- x - column name of the x coordinate in input file
- y - column name of the y coordinate in input file
'''
def get_grids(filename, outputFile = False, outputPath = None, n = 10, m = 10, x = 'x', y = 'y'):
    df = pd.read_csv(filename)
    
    # getting the cutoff values for x and y axis
    x_count, x_cutoff = np.histogram(df[x], bins = n)
    y_count, y_cutoff = np.histogram(df[y], bins = m)
    
    # transform cutoff values into step-wise tuples
    x_bin_tuple = [(floor, ceiling) for floor, ceiling in pairwise(x_cutoff)]
    y_bin_tuple = [(floor, ceiling) for floor, ceiling in pairwise(y_cutoff)]
    
    gridDict = {} # final output
    
    x_splits = split_df_rows_on_col_ranges(df, x, x_bin_tuple) # getting list of N bars based on x values
    
    # within each bar splitted based on x, there will be N splits based on y - each within which is a grid
    xidx = 0
    for xbar in x_splits:        
        # getting list of N bars (grids here already) based on y values, all within 1 xbar
        y_splits_in_xbar = split_df_rows_on_col_ranges(xbar, y, y_bin_tuple)
            
        yidx = 0
        for grid in y_splits_in_xbar:
            gridDict[(xidx, yidx)] = grid # gather output with x,y index
            yidx = yidx + 1     
        xidx = xidx + 1
        
    if outputFile:
        for key in gridDict:
            filename = 'x' + str(key[0]) + '_y' + str(key[1]) + '_grid.csv'
            fullpath = os.path.join(outputPath, filename)
            gridDict[key].to_csv(fullpath, index = False)
    else:
        return gridDict

'''
get_grids('/home/ec2-user/Kaggle/facebook_Jul_2016/input/train.csv',
          outputFile = True, outputPath = '/home/ec2-user/Kaggle/facebook_Jul_2016/input/10_10_grid/',
          n = 10, m = 10, x = 'x', y = 'y')
'''