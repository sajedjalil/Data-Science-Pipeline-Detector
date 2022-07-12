import numpy as np 
import pandas as pd



ids=[]

test = pd.read_csv("../input/test.csv")
ids = test["Id"]
mylist = []


for index, row in test.iterrows():
    line = [int(x) for x in row["Sequence"].split(',')]
    mylist.append(line)	




firstNo=[]
secondNo=[]

lastNo=[]
secondLastNo=[]
for i in range(len (mylist)):
    row = mylist[i] 
    row = list(map(int, row))
    firstNo.append(row[0])
    lastNo.append(row[-1])
    if (len(row)>1):
        secondNo.append(row[1])
        secondLastNo.append(row[-2])
    else:
        secondNo.append(row[0])
        secondLastNo.append(row[-1])
    

predictions=[]
count=0
for i, val in enumerate(firstNo):

    if (secondNo[i]==-99):
        predictions.append(firstNo[i])

    else:
        predictions.append(lastNo[i]+(lastNo[i]-secondLastNo[i]))



submission = pd.DataFrame({ 'Id': ids,
                            'Last': predictions })
submission.to_csv("submission.csv", index=False)


