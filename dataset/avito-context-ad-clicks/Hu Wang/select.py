#import pandas as pd
from csv import DictReader

#train_df = pd.read_csv('../input/trainSearchStream.tsv')
#num_contextads = len(train_df[train_df['ObjectType']==3])

train = '../input/trainSearchStream.tsv'

num_contextads = 0
num_ads = 0
for t, line in enumerate(DictReader(open(train), delimiter='\t')):
    if num_ads < 10:
        print(t, line)
    num_ads = num_ads + 1

#print('The number of context ads in the training data set is ', num_contextads, '(percentage is ', num_contextads*100.0/num_ads), ')')