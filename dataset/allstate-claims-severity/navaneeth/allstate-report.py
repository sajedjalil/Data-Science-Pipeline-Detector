######
######
######

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

# read csv file into dataframe
train_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")

# remove uncessary columns which are not usefull for predictions
train_set = train_set.drop(["id"],axis=1)
# preview data
#print(train_set.head())
#print(test_set.head())

# data info
#print(train_set.info())
#print(train_set.describe())

print(train_set["cont1"].mean())
print(test_set["cont1"].mean())
print(train_set["cont2"].mean())
print(test_set["cont2"].mean())
print(train_set["loss"].mean())

'''
parameters_train = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',\
                    'AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY',\
                    'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM','BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY',\
                    'CA','CB','CC','CD','CE','CF','CG','CH','CI','CJ','CK','CL','CM','CN','CO','CP','CQ','CR','CS','CT','CU','CV','CW','CX','CY',\
                    'DA','DB','DC','DD','DE','DF','DG','DH','DI','DJ','DK','DL','DM','DN','DO','DP','DQ','DR','DS','DT','DU','DV','DW','DX','DY',\
                    'EA','EB','EC','ED','EE','EF','EG','EH','EI','EJ','EK','EL','EM','EN','EO','EP','EQ','ES','EU','EV','EW','EY',\
                    'FA','FB','FC','FD','FE','FF','FG','FH','FI','FJ','FK','FL','FM','FN','FO','FP','FQ','FR','FS','FT','FU','FV','FW','FX',\
                    'GA','GB','GC','GD','GE','GF','GG','GH','GI','GJ','GK','GL','GM','GN','GO','GP','GQ','GR','GS','GT','GU','GV','GW','GX','GY',\
                    'HA','HB','HC','HD','HE','HF','HG','HH','HI','HJ','HK','HL','HM','HN','HO','HP','HQ','HR','HT','HU','HV','HW','HX','HY',\
                    'IA','IB','IC','ID','IE','IF','IG','IH','II','IJ','IK','IL','IM','IN','IO','IP','IQ','IR','IT','IU','IV','IX','IY',\
                    'JA','JB','JC','JD','JE','JF','JG','JH','JI','JJ','JK','JL','JM','JN','JO','JP','JQ','JR','JT','JU','JV','JW','JX','JY',\
                    'KA','KB','KC','KD','KE','KF','KG','KH','KI','KJ','KK','KL','KM','KN','KP','KQ','KR','KS','KT','KU','KV','KW','KX','KY',\
                    'LA','LB','LC','LD','LE','LF','LG','LH','LI','LJ','LK','LL','LM','LN','LO','LQ','LR','LT','LU','LV','LW','LX','LY',\
                    'MA','MB','MC','MD','ME','MF','MG','MH','MI','MJ','MK','ML','MM','MN','MO','MP','MQ','MR','MS','MT','MU','MV','MW','ZZ']
                     

value_train = 0
for i in parameters_train:
    value_train = value_train + 1
    train_set = train_set.replace(to_replace=i,value=value_train)

print(train_set.cat109[6429])

parameters_test = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',\
                   'AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY',\
                   'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM','BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY',\
                   'CA','CB','CC','CD','CE','CF','CG','CH','CI','CJ','CK','CL','CM','CN','CO','CP','CQ','CR','CS','CT','CU','CV','CW','CX','CY',\
                   'DA','DB','DC','DD','DE','DF','DG','DH','DI','DJ','DK','DL','DM','DN','DO','DP','DQ','DR','DS','DT','DU','DV','DW','DX','DY',\
                   'EA','EB','EC','ED','EE','EF','EG','EH','EI','EJ','EK','EL','EM','EN','EO','EP','ER','ES','ET','EU','EW','EX','EY',\
                   'FA','FB','FC','FD','FE','FF','FG','FH','FI','FJ','FK','FL','FM','FP','FQ','FR','FT','FU','FV','FW','FX','FY',\
                   'GA','GB','GC','GD','GE','GF','GG','GH','GI','GJ','GK','GL','GM','GN','GO','GP','GR','GS','GT','GU','GV','GW','GX','GY',\
                   'HA','HB','HC','HD','HE','HF','HG','HH','HI','HJ','HK','HL','HM','HN','HP','HQ','HR','HS','HT','HV','HW','HX','HY',\
                   'IA','IC','ID','IE','IF','IG','IH','II','IJ','IL','IM','IN','IP','IQ','IR','IS','IT','IU','IV','IW','IY',\
                   'JA','JB','JC','JE','JF','JG','JH','JJ','JK','JL','JM','JP','JQ','JR','JS','JU','JV','JW','JX','JY',\
                   'KA','KB','KC','KD','KE','KF','KG','KH','KI','KJ','KK','KL','KM','KN','KO','KP','KQ','KR','KS','KT','KU','KV','KW','KX','KY',\
                   'LA','LB','LC','LD','LE','LF','LG','LH','LI','LJ','LK','LL','LM','LN','LO','LP','LQ','LR','LS','LT','LU','LV','LW','LX','LY',\
                   'MA','MC','MD','ME','MG','MH','MI','MJ','MK','ML','MM','MN','MO','MP','MQ','MR','MS','MU','MV','MW','MX','ZZ']

value_test = 0
for i in parameters_test:
    value_test = value_test + 1
    test_set = test_set.replace(to_replace=i,value=value_test)

print(test_set.cat107[0]) 
'''
#sns.factorplot('cat1','loss',data=train_set)

'''
param_cont_train = ['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14']

for i in param_cont_train:
    train_set[i] = (train_set[i] * 1000).astype(int)
    
print(train_set["cont9"])

param_cont_test = ['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14']

for i in param_cont_test:
    test_set[i] = (test_set[i] * 1000).astype(int)
    
print(test_set["cont5"])
'''
### fit to the classifier
'''
X_train = train_set[["cont1","cont2","cont3","cont4","cont5","cont6","cont7","cont8","cont9","cont10","cont11","cont12","cont13","cont14"]].values
Y_train = train_set["loss"].values
Y_test = test_set[["cont1","cont2","cont3","cont4","cont5","cont6","cont7","cont8","cont9","cont10","cont11","cont12","cont13","cont14"]].values

classifier = linear_model.Lasso()
classifier.fit(X_train,Y_train)
print(classifier.score(X_train,Y_train))
Y_pred = classifier.predict(Y_test)

output = pd.DataFrame({"id":test_set["id"],"loss":Y_pred})
output.to_csv('output.csv',index=False)

'''








