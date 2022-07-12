# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins/"



def getProcess(start_with, ends_with):
    list_text = []
    list_file = []
    
    df_   = {}
    
    for filename in os.listdir(bulletin_dir):
        with open(bulletin_dir + "/" + filename, 'r', errors='ignore', encoding="ISO-8859-1") as f:
            
            content = f.read()
            content = content.replace("\n","").replace("\t","").replace(":","").strip() 
              
            text = find_between( content, start_with, ends_with )
            
            list_text.append( text )
            list_file.append( filename )
            
            df_[filename] = text
    
    return df_
    
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
         
def getJobTitle():    
    job_title = {}
    
    for filename in os.listdir(bulletin_dir):
        with open(bulletin_dir + "/" + filename, 'r', errors='ignore', encoding="ISO-8859-1") as f:
            job_title[filename] = f.readline()
            
    return job_title
            
def getOpenDate():
    openDate = {}
    
    # empty dict
    for i in os.listdir(bulletin_dir):
        openDate[i] = None
        
    for filename in os.listdir(bulletin_dir):
        with open(bulletin_dir + "/" + filename, 'r', errors='ignore', encoding="ISO-8859-1") as f:
             for line in f.readlines():
                 if line.__contains__( 'Open Date:'):
                    openDate[filename]= line.split()[2]
                    
    return openDate

def exctractCode(x):
    x = x.split()[-1:]
    str1 = ''.join(x)
    return str1
    
    
    
# get job title
df_dict                      = getJobTitle()
df__                         = pd.DataFrame( )
df__['FILE_NAME']            = df_dict.keys() 
df__['JOB_CLASS_TITLE']      = df_dict.values()

# get class_code
start_with                   = ' '
ends_with                    = 'Open Date'
df_dict                      = getProcess(start_with, ends_with)
df__['JOB_CLASS_NO']         = df_dict.values()
#df__['JOB_CLASS_NO']         = df__['JOB_CLASS_NO'].apply(lambda x: x.split()[-1:] )
df__['JOB_CLASS_NO']         = df__['JOB_CLASS_NO'].apply( exctractCode )

# get Open Date
start_with                   = 'Class Code'
ends_with                    = 'ANNUAL SALARY'
df_dict                      = getOpenDate()
df__['OPEN_DATE']            = df_dict.values()


# get SALARY
start_with                   = 'ANNUAL SALARY'
ends_with                    = 'NOTE'
df_dict                      = getProcess(start_with, ends_with)
df__['ANNUAL_SALARY']        = df_dict.values()

    
# get NOTE
start_with                   = 'NOTE'
ends_with                    = 'DUTIES'
df_dict                      = getProcess(start_with, ends_with)
df__['NOTE']                 = df_dict.values()

# get DUTIES
start_with                   = 'DUTIES'
ends_with                    = 'QUALIFICATION'#'REQUIREMENT/MINIMUM QUALIFICATION'
df_dict                      = getProcess(start_with, ends_with)
df__['DUTIES']               = df_dict.values()

# get 'REQUIREMENT/MINIMUM QUALIFICATION'
start_with                   = 'QUALIFICATION'#'REQUIREMENT/MINIMUM QUALIFICATION'
ends_with                    = 'PROCESS NOTES'
df_dict                      = getProcess(start_with, ends_with)
df__['REQUIREMENT_MIN_QUAL'] = df_dict.values()


# get PROCESS NOTES
start_with                   = 'PROCESS NOTES'
ends_with                    = 'WHERE TO APPLY'
df_dict                      = getProcess(start_with, ends_with)
df__['PROCESS_NOTE']         = df_dict.values()


# get WHERE TO APPLY
start_with                   = 'WHERE TO APPLY'
ends_with                    = 'NOTE'
df_dict                      =  getProcess(start_with, ends_with)
df__['WHERE_TO_APPLY']       = df_dict.values()


#get NOTE:
start_with                   = 'NOTE'
ends_with                    = 'APPLICATION DEADLINE'
df_dict                      =  getProcess(start_with, ends_with)
df__['NOTE']                 = df_dict.values()

#get APPLICATION DEADLINE
start_with                   = 'APPLICATION DEADLINE'
ends_with                    = 'SELECTION PROCESS'
df_dict                      =  getProcess(start_with, ends_with)
df__['APPLICATION_DEADLINE'] = df_dict.values()


#get 'SELECTION PROCESS'
start_with                   = 'SELECTION PROCESS'
ends_with                    = 'NOTICE'
df_dict                      =  getProcess(start_with, ends_with)
df__['SELECTION_PROCESS']    = df_dict.values()

#get ' NOTICE'
start_with                   = 'NOTICE'
ends_with                    = 'AN EQUAL EMPLOYMENT OPPORTUNITY EMPLOYER'
df_dict                      =  getProcess(start_with, ends_with)
df__['NOTICE']               = df_dict.values()
