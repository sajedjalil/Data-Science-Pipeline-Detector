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



train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])#,usecols=train_cols)
test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])#,usecols=test_cols)


train.drop(['id'],axis=1, inplace=True)
test.drop(['id'],axis=1, inplace=True)


def datasource_compare(datasource, compareto):
    
    numeric_cols = []
    str_cols = []
    
    if (not datasource.empty):
        
        for c in datasource.columns.values:    
            
            if ((c in datasource.columns) and (c in compareto.columns)):
        
                if (datasource[c].dtype == 'int64') or (datasource[c].dtype == 'float64'):
                    numeric_cols.append(c)
                if (datasource[c].dtype == 'object'):
                    str_cols.append(c)
                
        for nc in numeric_cols:
            print("{:>40} {:>10} {:>10} {:15} {:15} {:15} {:15} {:15} {:15}".format(nc, str(datasource[nc].dtype), str(compareto[nc].dtype), len(datasource[nc]), len(compareto[nc]), datasource[nc].count(), compareto[nc].count(), len(datasource[datasource[nc].isnull()]), len(compareto[compareto[nc].isnull()])))
            print("{:62} {:15.2f} {:15.2f} {:15.2f} {:15.2f} {:15.2f} {:15.2f} {:15.2f} {:15.2f}".format(" ",datasource[nc].mean(), compareto[nc].mean(), datasource[nc].min(), compareto[nc].min(), datasource[nc].max(), compareto[nc].max(), datasource[nc].std(),  compareto[nc].std() ))
            print("{:62} {:15.2f} {:15.2f} {:15.2f} {:15.2f} {:15.2f} {:15.2f}".format(' ',datasource[nc].quantile(.25),compareto[nc].quantile(.25),datasource[nc].quantile(.50),compareto[nc].quantile(.50),datasource[nc].quantile(.75),compareto[nc].quantile(.75)))
            print()
        print ("\n\n")
            
        for sc in str_cols:
     
           # print(datasource[sc].sc.count())
                    
           unique=datasource[sc].unique()
           unique_compareto = compareto[sc].unique()
                    
           group_counts = datasource.groupby(sc).size()
           group_counts_compareto = compareto.groupby(sc).size()
           
           total_count = group_counts.sum()
           total_count_compareto = group_counts_compareto.sum()
           
           print ("\n")
                                       
           print("{:>40} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}".format(sc, len(unique),len(unique_compareto), group_counts.min(), group_counts_compareto.min(), group_counts.max(), group_counts_compareto.max(), len(datasource[datasource[sc].isnull()]), len(compareto[compareto[sc].isnull()])))
               
           print("")
           for index in group_counts.keys():              
               if index in group_counts_compareto.index:                       
                   print("{:>45} {:10} {:10} {:10.2f}% {:10.2f}%".format(index, group_counts.get_value(index), group_counts_compareto.get_value(index), (group_counts.get_value(index)/total_count)*100, (group_counts_compareto.get_value(index)/total_count_compareto)*100))
               else:
                   print("{:>45} {:10} {:10} {:10.2f}%".format(index, group_counts.get_value(index), 0, (group_counts.get_value(index)/total_count)*100))

                        
    else:
        print ("sorry, no data present")
          
   
datasource_compare(train,test)        