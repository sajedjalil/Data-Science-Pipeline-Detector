
import os
print(os.listdir("../input"))
from csv import DictReader
import pandas as pd
import numpy as np

## column groups
cat_features = ['ip', 'app', 'device', 'os', 'channel']

wset=  'train'
print('producing ' + wset)
loc_file = '../input/'+wset+'.csv'
loc_file_vw = wset+'.vw'


with open(loc_file_vw,"w") as outfile:
    for linenr, row in enumerate( DictReader(open(loc_file,"r")) ):
        # initialize empty namespaces
        n_c = ''

        for k in row:
            if k in cat_features:
                n_c += ' %s_%s'%(k[0:3],str(row[k]))
            elif k == 'click_time':
                xhour = row['click_time'][11:13]
                n_c += ' %s_%s'%('hour',str(xhour))


        # prepare others
        #id = str(row['id'])
        id = str(linenr)
        label = str(2 * int(row['is_attributed']) - 1 )

        outfile.write("%s '%s |c%s \n"%(label,id,n_c) )
