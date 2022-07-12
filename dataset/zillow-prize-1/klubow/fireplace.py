# -*- coding: utf-8 -*-
import pandas as pd

if __name__ == '__main__':

    print('Loading data ...')
    pr = pd.read_csv('../input/properties_2016.csv', low_memory=True)
    print('After loading ', pr.shape)  
    
    
    #no duplicates, not even in parcelid
#    pr = pr.drop_duplicates(subset=None, keep='first')    
#    print('After drop_duplicates ', pr.shape)  
    
    #no all nans cols
#    pr = pr.dropna(axis=1, how="all", thresh=None, subset=None, inplace=False)    
    
    
    
    
    pr['fireplaceflag'] = pr['fireplaceflag'].fillna(value=-1)   
    pr['fireplacecnt'] = pr['fireplacecnt'].fillna(value=-1)      


    print(pr['fireplaceflag'].value_counts())
    print(pr['fireplacecnt'].value_counts())
    
