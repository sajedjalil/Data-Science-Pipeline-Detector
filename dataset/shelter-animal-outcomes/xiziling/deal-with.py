import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import cross_val_score

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#in order to save time
train = train[0: 10000]

L1 = train.values[:,1] #name
L2 = train.values[:,2] #datetime

L3 = train.values[:,3] #out
L4 = train.values[:,4] #
L5 = train.values[:,5] #type
L6 = train.values[:,6] #sex
L7 = train.values[:,7] #age
L8 = train.values[:,8] #breed
L9 = train.values[:,9] #color

pro_data = list()

for L in [L3,L5,L6,L8,L9]:
    l = list(set(L))
    l=np.array(l)
    
    for s in l:   
        split = [1 if x == s else x for x in L]
        split = [0 if x != 1 else x for x in split]
        #print s
        split = np.array(split)
        pro_data.append(split)
b=list()
for a in L7:
    if type(a) == float:
        a=["0","0"]
    else:      
        a = a.split(' ')
        a=[a[0],a[1].replace('years','365')]
        a=[a[0],a[1].replace('year','365')]
        a=[a[0],a[1].replace('months','30')]
        a=[a[0],a[1].replace('month','30')]
        a=[a[0],a[1].replace('weeks','7')]
        a=[a[0],a[1].replace('week','7')]
        a=[a[0],a[1].replace('days','1')]
        a=[a[0],a[1].replace('day','1')]

    c= eval(a[0])*eval(a[1])
    b.append(c)
   
pro_data.append(b)    
#----------time process-->>>-----------------------
d1=list()
d2=list()
d3=list()
d4=list()
d5=list()
for a in L2:
    a=a.split(r'-')
    b=a[2].split(' ')
    c=b[1].split(':')
    d1.append(int(a[0]))
    d2.append(int(a[1]))
    d3.append(int(b[0]))
    d4.append(int(c[0]))
    d5.append(int(c[1]))
    
for dx in (d1,d2,d3,d4,d5):
    
    pro_data.append(dx)
#----------time process--<<<<-----------------------
vectorizer = TfidfVectorizer()    
XL1 = ['0' if type(x) == float else x for x in L1]
XL1 = vectorizer.fit_transform(XL1)
XL1=XL1.toarray()
pro_data = np.array(pro_data)
pro_data = pro_data.transpose()
pro_data=np.hstack((pro_data,XL1))
pro_data = np.array(pro_data)

x_train = pro_data[:,5:]
y_train = pro_data[:,:5]

mds =[10,20,30,40,50,60,70,90,120,160,200]
scores=list()
for md in mds:
    dtc = DecisionTreeClassifier(max_depth=md)
    score= cross_val_score(dtc,x_train,y_train,cv=10)
    scores.append(score.mean())
    print ("max_depth %s finished" % md)

plt.plot(mds,scores,'b-o')
plt.xlabel('max_depths')
plt.ylabel('scores')
plt.title('10000 samples used')
plt.savefig('1.png')
plt.show()
#----------submition--->>>>>>--------------
c_clf = DecisionTreeClassifier(max_depth=40)
c_clf.fit(x_train,y_train)
#use training data
preds = c_clf.predict(pro_data[0:100,5:])
pred_label1=list()
pred_label2=list()
pred_label3=list()
pred_label4=list()
pred_label5=list()

for pred in preds:
    if list(pred)==[1, 0, 0, 0, 0]:
        #pred_label1.('Transfer')
        pred_label1.append('0')
        pred_label2.append('0')
        pred_label3.append('0')
        pred_label4.append('0')
        pred_label5.append('1')
    elif list(pred)==[0, 1, 0, 0, 0]:
        #pred_label2.('Adoption')
        pred_label1.append('1')
        pred_label2.append('0')
        pred_label3.append('0')
        pred_label4.append('0')
        pred_label5.append('0')
    elif list(pred)==[0, 0, 1, 0, 0]:
        #pred_label3.'Return_to_owner')
        pred_label1.append('0')
        pred_label2.append('0')
        pred_label3.append('0')
        pred_label4.append('1')
        pred_label5.append('0')
    elif list(pred)==[0, 0, 0, 1, 0]:
        #pred_label4.('Died')
        pred_label1.append('0')
        pred_label2.append('1')
        pred_label3.append('0')
        pred_label4.append('0')
        pred_label5.append('0')
    else:	    
       #pred_label5.('Euthanasia')
        pred_label1.append('0')
        pred_label2.append('0')
        pred_label3.append('1')
        pred_label4.append('0')
        pred_label5.append('0')

submission = pd.DataFrame({"Adoption": pred_label1,"Died":pred_label2,"Euthanasia":pred_label3,"Return_to_owner":pred_label4,"Transfer":pred_label5})
submission.insert(0, 'ID', range(1,101))
submission.to_csv("submission.csv", index=False)
