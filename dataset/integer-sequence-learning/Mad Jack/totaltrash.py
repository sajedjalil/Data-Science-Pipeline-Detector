import numpy as np 
from scipy.optimize import curve_fit
import pandas as pd
import math

def mode(arr) :
    m = max([arr.count(a) for a in arr])
    return [x for x in arr if arr.count(x) == m][0] if m>1 else None


ids=[]

test = pd.read_csv("../input/test.csv")
ids = test["Id"]
mylist = []

def func(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d
    
def func2(x, a, b, c):
    return a * np.exp(-b * x) + c

for index, row in test.iterrows():
    line = [int(x) for x in row["Sequence"].split(',')]
    mylist.append(line)	
    
firstNo=[]
secondNo=[]
linearFit = True
lastNo=[]
secondLastNo=[]
thirdLastNo=[]
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
        secondLastNo.append(row[0])
    if (len(row)>2):
        thirdLastNo.append(row[-3])
    else:
        thirdLastNo.append(row[0])

predictions=[]
for i in range(len (mylist)):
    linearFit = False
    row = mylist[i] 
    row = list(map(float, row))
    a = np.array(row)
    x = np.arange(0,len(a))


    m = mode(row)
    if (len(a)==1):
        predictions.append(firstNo[i])


    elif (len(a)<=3): 
        predictions.append(lastNo[i]+(lastNo[i]-secondLastNo[i]))
        linearFit = True
    else:
        try:
            popt, pcov = curve_fit(func, x, a)
            g = ((len(a)))
            check = 0
            
            for q in range(len(a)):
                check = check + abs((popt[0]*q**3 + popt[1]*q**2 + popt[2]*q + popt[3])-a[q])

            if (math.fabs(check)<10):			
                predictions.append(int(popt[0]*g**3 + popt[1]*g**2 + popt[2]*g + popt[3]))
            else:
                try:
                    popt, pcov = curve_fit(func2, x, a)
                    g = ((len(a)))
                    check = 0
            
                    for q in range(len(a)):
                        check = check + abs((popt[0]*np.exp(-popt[1]*q)+popt[2])-a[q])
                    if (math.fabs(check)<10):			
                        predictions.append(int(popt[0] *np.exp(-popt[1]*g) + popt[2]))
                    else:
                        predictions.append(lastNo[i]+(lastNo[i]-secondLastNo[i]))
                        linearFit = True
                except RuntimeError:
                    predictions.append(lastNo[i]+(lastNo[i]-secondLastNo[i])) 
                    linearFit = True
        except RuntimeError:
            predictions.append(lastNo[i]+(lastNo[i]-secondLastNo[i]))
            linearFit = True
    if (math.fabs((secondLastNo[i]-thirdLastNo[i])-(lastNo[i]-secondLastNo[i]))>10) and linearFit and m!=None:
        del predictions[-1]


        predictions.append(int(m))


submission = pd.DataFrame({ 'Id': ids,
                            'Last': predictions })
submission.to_csv("submission.csv", index=False)