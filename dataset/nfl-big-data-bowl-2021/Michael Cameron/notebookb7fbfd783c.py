# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import glob
df = pd.concat([pd.read_csv(f, encoding='latin1') for f in glob.glob('../input/nfl-big-data-bowl-2021/week*.csv')], ignore_index=True)
plays = pd.read_csv("../input/nfl-big-data-bowl-2021/plays.csv")

def distance(x,y):
    dist = np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2) )
    return dist

final_ball = df[(df['event'].isin(['pass_outcome_caught','pass_outcome_incomplete','pass_outcome_interception']))&(df['displayName']=='Football')]
final_ball.drop_duplicates(subset = ['gameId', 'playId'], keep = 'first', inplace = True)

final_offence = df[(df['event'].isin(['pass_outcome_caught','pass_outcome_incomplete','pass_outcome_interception']))&(df['route'].notnull())]
final_offence.drop_duplicates(subset = ['nflId','gameId', 'playId'], keep = 'first', inplace = True)


join_df = final_offence.merge(final_ball[['x','y','gameId','playId']], how = 'left', on = ['gameId', 'playId'], suffixes=['_off', '_ball'])

join_df['ball_dist'] = join_df.apply(lambda row: distance([row['x_off'], row['y_off']], [row['x_ball'],row['y_ball']]), axis=1)

join_df['min_dist'] = join_df.groupby(['gameId', 'playId'])['ball_dist'].transform('min')

def target(dist_to_ball, minimum_dist):
    if (dist_to_ball == minimum_dist) & (dist_to_ball <20):
        return 1
    else:
        return 0

join_df['target'] = join_df.apply(lambda row: target(row['ball_dist'],row['min_dist']), axis=1)

targets = join_df[['displayName','target','gameId','playId', 'nflId']]

targets.to_csv('/kaggle/working/targets.csv',index=False)