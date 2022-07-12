import numpy as np,pandas as pd,random


def load_data_file(file_name,names):
    #if 'test' in file_name:
    #    return shuffle_dataframe(pd.read_csv(file_name, sep='\t', engine='python', encoding='utf-8').rename(columns=names))
    return pd.read_csv(file_name, sep='\t', engine='python', encoding='utf-8').rename(columns=names)

def shuffle_strings(x):
    return x.apply(lambda x:''.join(random.sample(repr(x),len(repr(x)))))

def shuffle_dataframe(data):
    data['brand_name']=shuffle_strings(data['brand_name'])
    data['category_name']=shuffle_strings(data['category_name'])
    data['name']=shuffle_strings(data['name'])
    data['item_description']=shuffle_strings(data['item_description'])
    return data

path='../input/'
train=load_data_file(path+'train.tsv',{'train_id':'id'})
test=pd.concat([load_data_file(path+'test.tsv',{'test_id':'id'}),
                #shuffle_dataframe(load_data_file(path+'test.tsv',{'test_id':'id'})),
                load_data_file(path+'test.tsv',{'test_id':'id'}),
                load_data_file(path+'test.tsv',{'test_id':'id'}),
                shuffle_dataframe(load_data_file(path+'test.tsv',{'test_id':'id'})),
                #               ^
                #               |
                #      This is shuffled data, 
                #   sames only can add one set, which allow my rest part of code went over
                # it may be cuased by the shuffle is to much 
                load_data_file(path+'test.tsv',{'test_id':'id'})])
data=pd.concat([train,test])
del train,test
#data=pd.concat([load_data_file(path+'train.tsv',{'train_id':'id'}),
#                load_data_file(path+'test.tsv',{'test_id':'id'})])
sep=data['price'].notnull().sum()
predict=data[data['price'].isnull()][['id']].rename(columns={'id':'test_id'})
train_y=np.float32(np.log1p(data[data['price'].notnull()]['price']))
