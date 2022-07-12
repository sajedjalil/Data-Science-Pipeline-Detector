from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
data.drop(columns=["Id","Province_State"],inplace=True)
xshape,yshape=data.shape
for i in range(xshape):
    day=0
    temp1=str(data.iloc[i,1])
    temp=temp1.split("-")
    a=int(temp[1])
    if(a==1):
        day=day+int(temp[2])
    elif(a==2):
        day=day+int(temp[2])+31
    elif(a==3):
        day=day+int(temp[2])+60
    elif(a==4):
        day=day+int(temp[2])+91
    elif(a==5):
        day=day+int(temp[2])+121
    else:
        day=day+int(temp[2])+152
    data.iloc[i,1]=day
data["Country_Region"].nunique()
country=[]
for i in range(xshape):
    if(data.iloc[i,0] not in country):
        country.append(data.iloc[i,0])   
casepredmodel=[]
fatalitypredmodel=[]
count=0
for i in country:
    d=[]
    start=0
    end=0
    flag=0
    for j in range(xshape-1):
        if( (data.iloc[j,0]==i) and flag==0):
            start=j
            flag=1
        if(data.iloc[j,0]==i):
            end=j
    d=data.iloc[start:end].copy()
    polynomial_features= PolynomialFeatures(degree=3)

    casemodel=LinearRegression().fit(polynomial_features.fit_transform(pd.DataFrame(d["Date"])),d["ConfirmedCases"])
    fatalitymodel=LinearRegression().fit(polynomial_features.fit_transform( pd.DataFrame(d["Date"]),pd.DataFrame(d["ConfirmedCases"]) ),d["Fatalities"])
    casepredmodel.append(casemodel)
    fatalitypredmodel.append(fatalitymodel)
    count+=1
    print(count,sep=" ")

test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
test=test.drop(columns=["ForecastId","Province_State"],axis=1)
testx,testy=test.shape
for i in range(testx):
    day=0
    temp1=str(test.iloc[i,1])
    temp=temp1.split("-")
    a=int(temp[1])
    if(a==1):
        day=day+int(temp[2])
    elif(a==2):
        day=day+int(temp[2])+31
    elif(a==3):
        day=day+int(temp[2])+60
    elif(a==4):
        day=day+int(temp[2])+91
    elif(a==5):
        day=day+int(temp[2])+121
    elif(a==6):
        day=day+int(temp[2])+152
    elif(a==7):
        day=day+int(temp[2])+182
    elif(a==8):
        day=day+int(temp[2])+213
    elif(a==9):
        day=day+int(temp[2])+243
    elif(a==10):
        day=day+int(temp[2])+274
    else:
        day=day+int(temp[2])+304
    test.iloc[i,1]=day
casepred=[]
fatpred=[]
for i in range(testx): #testx
    b=test.iloc[i,0]
    index=country.index(b)
    mod1=casepredmodel[index]
    mod2=fatalitypredmodel[index]
    
    polynomial_features= PolynomialFeatures(degree=3)
    temp0=test.iloc[i,1]
    print(temp0)
    temp=polynomial_features.fit_transform([[temp0]])
    
    m=int(mod1.predict(temp))
    casepred.append(m)
    temp1=polynomial_features.fit_transform([[temp0]],[[m]])
    fatpred.append(int(mod2.predict(temp1)))
sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
sub["ConfirmedCases"]=casepred
sub["Fatalities"]=fatpred
sub.to_csv("submission.csv",index=False)


