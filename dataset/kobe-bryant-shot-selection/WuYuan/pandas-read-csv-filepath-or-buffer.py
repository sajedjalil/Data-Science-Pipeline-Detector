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

df = pd.pandas.read_csv("../input/data.csv")
length = len(df)
print(length)


#print(df)

#pre processing of data
game_event_id_list = df['game_event_id'].unique()
#print(game_event_list)

for value in game_event_id_list:
#    print(value)
    df.loc[:,value] = pd.Series(1.0, index=df.index)
#    df.loc[:,value] = pd.Series(df.loc['game_event'] == value, index=df.index)
    
print(df)

action_type_list = df['action_type'].unique()

#for value in action_type_list:
#    df.loc[:,value] = pd.Series(np.random.randn(), index=df.index)
    
    
    



df_t = df[np.logical_not(np.isnan(df['shot_made_flag']))]
df_p = df[np.isnan(df['shot_made_flag'])]


