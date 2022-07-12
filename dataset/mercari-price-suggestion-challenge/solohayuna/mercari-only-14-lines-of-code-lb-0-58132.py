import pandas as pd
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
for c in ['name', 'category_name', 'brand_name', 'item_description']:
    train[c] = train[c].fillna('no_'+c).str.lower()
    test[c] = test[c].fillna('no_'+c).str.lower()
cols = ['item_description', 'name', 'brand_name', 'category_name', 'item_condition_id', 'shipping']
for i in range(0,6):
    test = test.merge(train.groupby(cols[i:]).agg({'price': 'mean'}).reset_index(), 
                      on=cols[i:], how='left', suffixes=['', '_{}'.format(i)])
for i in range(1,6):
    test.loc[test['price'].isnull(), 'price'] = test['price_{}'.format(i)]
test.loc[test['price'].isnull(), 'price'] = train['price'].mean()
test[['test_id', 'price']].to_csv('sub_just_means.csv', index=False)