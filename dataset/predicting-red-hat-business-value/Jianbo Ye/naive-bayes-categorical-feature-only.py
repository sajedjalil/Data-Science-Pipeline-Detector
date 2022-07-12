import pandas as pd
import numpy as np

people=pd.read_csv('../input/people.csv')
#people=people.replace({'type ': '', 'group ': '', 'ppl_': ''}, regex=True)
act_train=pd.read_csv('../input/act_train.csv')
#act_train=act_train.replace({'type ': '', 'group ': '', 'ppl_': ''}, regex=True)
act_train=act_train.fillna('-1')
act_test=pd.read_csv('../input/act_test.csv')
#act_test=act_test.replace({'type ': '', 'group ': '', 'ppl_': ''}, regex=True)
act_test=act_test.fillna('-1')

joined_data=pd.merge(people, act_train, on='people_id')
joined_test=pd.merge(people, act_test, on='people_id')

a=dict()
b=dict()
for name in joined_data.columns.values:
    if name != 'people_id' and name != 'activity_id' and name != 'date_x' and name != 'date_y' and name != 'char_38':
        print(name)
        a[name]=joined_data[joined_data['outcome'] == 0][name]
        a[name]=a[name].append(pd.DataFrame({0:joined_data[name].unique()}))[0]
        a[name]=-np.log(a[name].value_counts(dropna=False)/len(a[name]))
        a[name]=dict(a[name])
        b[name]=joined_data[joined_data['outcome'] == 1][name]
        b[name]=b[name].append(pd.DataFrame({0:joined_data[name].unique()}))[0]
        b[name]=-np.log(b[name].value_counts(dropna=False)/len(b[name]))
        b[name]=dict(b[name])
        
aa=np.zeros(len(joined_test))
bb=np.zeros(len(joined_test))
for name in joined_test.columns.values:
    if name != 'people_id' and name != 'activity_id' and name != 'date_x' and name != 'date_y' and \
       name != 'char_38' and name != 'group_1' and name != 'char_10_y':
        print(name)
        aa = aa + np.array([a[name][x] for x in joined_test[name]])
        bb = bb + np.array([b[name][x] for x in joined_test[name]])

submission=pd.DataFrame({'activity_id':act_test['activity_id'], 'outcome':np.argmin(np.concatenate((aa, bb)).reshape((2, len(aa))).T, 1)})
submission.to_csv('submit.csv', index=False)

