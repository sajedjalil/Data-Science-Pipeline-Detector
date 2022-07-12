'''
Blends your submission files

Just change mypath to your path with files
There should be only .CSV submission files in that directory
'''

import pandas as pd
import os

def blend(files, path = '', id_name = 'Id', target_column_name = 'Sales'):
    if len(files) < 2:
        raise ValueError('There should be at least 2 submission files to blend')
    res = pd.read_csv(path + files[0])
    for i, filename in enumerate(files):
        print 'Reading %s' % filename
        if i == 0: continue
        sample = pd.read_csv(path + filename)
        res = pd.concat([res, sample[target_column_name]], axis=1)
    print 'Averaging data...'
    res.drop(id_name, axis=1, inplace=True)
    sample[target_column_name] = res.mean(axis=1)
    sample.to_csv(path + 'submission_blended.csv', index=False)
    print 'Done'

if __name__ == '__main__':
    mypath = 'C:\\Kaggle\\Rossman\\submissions\\'
    files = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
    print files
    blend(files, mypath)