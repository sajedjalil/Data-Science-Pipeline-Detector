"""
=================================================
Data exploration
=================================================

Extract fields from 'page' variable: (lang, access, name, agent),
and store them in a separated variables, where the article name has the following fomrat:
'name_project_access_agent'

Developed by Muhanad Shab Kaleia: ms.kaleia@gmail.com
"""
print(__doc__)

import pandas as pd
import os.path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Read the training set
train_1_path = '../input/train_1.csv'
df = pd.DataFrame.from_csv(train_1_path, index_col=None, encoding='utf-8')

# Add new fields, lang, article name, access, agent
df['agent'] = map(lambda v:v.split('_')[-1], df.Page)
df['access'] = map(lambda v:v.split('_')[-2], df.Page)
df['lang'] = map(lambda v: v.split('_')[-3].split('.')[0], df.Page)
df['article_name'] = map(lambda v: v.split('_')[0], df.Page)

# Wtite the output into a csv file
# df.to_csv(BASE_DIR + '/dataset.csv', encoding='utf-8')
