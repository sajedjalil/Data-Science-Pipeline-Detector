_author_ = 'sonu'
import pandas as pd
import numpy as np
import gzip,csv
train = pd.read_csv(r'../input/train.csv')
N = float(len(train))
DaysOfWeek = sorted(np.unique(train['DayOfWeek']))
PdDistricts = sorted(np.unique(train['PdDistrict']))
classes = sorted(np.unique(train['Category']))
Sol =['Id']
Sol += classes
trainS = train[['DayOfWeek','PdDistrict','Category']]

dictDay = {}
dictCat = {}
dictDayWithCat={}

for i in DaysOfWeek:
    for j in classes:
        selected = trainS[ trainS['Category']==j][['DayOfWeek']]
        selectedDay = len(trainS[trainS['DayOfWeek']==i][['DayOfWeek']])
        if len(selected)!=0:
            selectedWithDay = selected [ selected['DayOfWeek']==i]['DayOfWeek']
            probDbyC = len(selectedWithDay)/float(len(selected))
            probD = selectedDay/N
            probC = len(selected)/N
            dictDay[i]=probD
            dictCat[j]=probC
            dictDayWithCat[(i,j)]=probDbyC

dictDis = {}
dictDisWithCat={}

for i in PdDistricts:
    for j in classes:
        selected = trainS[ trainS['Category']==j][['PdDistrict']]
        selectedDis = len(trainS[trainS['PdDistrict']==i][['PdDistrict']])
        if len(selected)!=0:
            selectedWithDis = selected [ selected['PdDistrict']==i]['PdDistrict']
            probDbyC = len(selectedWithDis)/float(len(selected))
            probD = selectedDis/N
            dictDis[i]=probD
            dictDisWithCat[(i,j)]=probDbyC

#print len(dictDis),len(dictDisWithCat)

test = pd.read_csv(r'../input/test.csv')
outf = gzip.open(r'Output.csv.gz','wt')
fo = csv.writer(outf, lineterminator='\n')
fo.writerow(Sol)
for (idd,dist,day) in zip(test['Id'],test['PdDistrict'],test['DayOfWeek']):
    sol = [idd]
    for _class in classes:
        ans1 = dictDisWithCat[(dist,_class)] * dictDayWithCat[(day,_class)] * dictCat[_class]
        ans2 = dictDay[day] * dictDis[dist]
        ans = ans1/ans2
        sol.append(ans)
    fo.writerow(sol)
