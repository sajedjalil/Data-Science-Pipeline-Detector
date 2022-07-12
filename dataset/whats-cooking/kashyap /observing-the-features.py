



def main():
    pass

if __name__ == '__main__': 
    main()

import json
from pprint import pprint
from scipy.stats import itemfreq

with open('../input/train.json') as data_file:
    train_data_json = json.load(data_file)

with open('../input/test.json') as data_file:
    test_data_json=json.load(data_file)

#to observe what all ingredients are there in all the foods
train_ingredient_all=[]
for i in range(0,len(train_data_json)):
#for i in range(0,500):
    for j in range(0,len(train_data_json[i]["ingredients"])):
        train_ingredient_all.append((str((train_data_json[i]["ingredients"][j]).encode('ascii','ignore'))))

train_ingredient_uniq=itemfreq(train_ingredient_all)
train_ingredient_uniq=train_ingredient_uniq[:,0]




