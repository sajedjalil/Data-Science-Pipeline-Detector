import pandas as pd
import nltk
df_cooking = pd.read_csv('../input/cooking.csv')
df_bio = pd.read_csv('../input/biology.csv')
df_crypto = pd.read_csv('../input/crypto.csv')
df_diy = pd.read_csv('../input/diy.csv')
df_robotics = pd.read_csv('../input/robotics.csv')
df_travel = pd.read_csv('../input/travel.csv')

print (df_diy.head())