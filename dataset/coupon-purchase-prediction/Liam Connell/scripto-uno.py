import os
import glob
import re
import pandas as pd
os.system("ls ../input")
#os.system("echo \n\n")
#os.system("head ../input/*")



#os.system('../input/*')
datadict = {}
for file in glob.glob('../input/*.csv'):
    datadict[re.split('/', file)[2]] = pd.read_csv(file)
    print(datadict[re.split('/', file)[2]].head())
    