_author_ = 'sonu'
import pandas as pd
import numpy as np
import gzip,csv
import matplotlib.pyplot as plt
train = pd.read_csv(r'../input/train.csv')

def getOnlyTime(a):
    a=a.split(' ')
    a=a[1].split(':')
    a=a[0]
    return int(a)

classes = sorted(np.unique(train['Category']))
Sol =['Id']
Sol += classes

y= train.Category.values
train = train.Dates.values

train_times = map(getOnlyTime,train)

ar =[0]*24
dict = {}
for _class in classes:
    dict[_class] = len(dict)
    
val = np.zeros((25,len(classes)),dtype=float)

for (i,j) in zip(train_times,y):
    ar[i]+=1
    val[i,dict[j]]+=1

    
print(val)

test = pd.read_csv(r'../input/test.csv')
test_times = map(getOnlyTime,test.Dates.values) 
ids = test['Id']
outf = gzip.open(r'Output.csv.gz','wt')
fo = csv.writer(outf, lineterminator='\n')
fo.writerow(Sol)
for (id,time) in zip(ids,test_times):
    sol=[id]
    for i in classes:
        ans = val[time][dict[i]]/ar[time]
        sol.append(ans)
    fo.writerow(sol)
'''x=range(1,25)
plt.plot(x,ar)
plt.savefig('time.png')'''