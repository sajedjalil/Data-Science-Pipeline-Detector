#This file takes the test and train csv and outputs a newtest and newtrain csv with new columns
#takes about 50 min to run
#final number of columns is 2116
#The output is too large to run as a script (need to make a data train)
#exceeds memory limts for scipts (when running delete nrows=nrows from read_csv)




import numpy as np
import pandas as pd
import datetime

#used to create indicator variables
def ind99(x):
  return (x==999999999)*1
def ind98(x):
  return (x==999999998)*1
def ind97(x):
  return (x==999999997)*1
def ind96(x):
  return (x==999999996)*1
def ind95(x):
  return (x==999999995)*1
def ind94(x):
  return (x==999999994)*1
def makesmall(x):
  if x>999999900:
    return np.nan
  else:
    return x
def toIndicate(x,value):
  return (x==value)*1

#Columns with timestamps
# From manual data analysis
dtCol=[73,75,156,157,158,159,166,167,168,169,176,177,178,179,204,217]

#numerical columns use 999999999,999999998,999999997,999999996,999999995,999999994 as an indicator.
col99=[541,912]
col98=[575, 583, 598, 607, 641, 648, 659, 669, 670, 673, 691, 698, 910, 919, 946, 950, 964, 968, 1179, 1199, 1215, 1227, 1241, 1353, 1398, 1418, 1451, 1715, 1730, 1747, 1754, 1801, 1845, 1859, 1869, 1912, 1922, 1929]
col97=[541, 575, 584, 598, 607, 641, 648, 669, 670, 673, 691, 910, 912, 917, 918, 919, 944, 946, 950, 964, 970, 1086, 1179, 1199, 1215, 1227, 1241, 1353, 1398, 1418, 1451, 1617, 1687, 1688, 1689, 1715, 1730, 1747, 1754, 1801, 1845, 1859, 1869, 1912, 1922, 1929]
col96=[541, 544, 575, 576, 583, 584, 586, 598, 600, 607, 608, 610, 641, 642, 643, 648, 649, 651, 659, 669, 670, 673, 674, 691, 698, 944, 950, 964, 968, 970, 976, 977, 1083, 1179, 1180, 1199, 1215, 1227, 1228, 1241, 1243, 1244, 1248, 1312, 1341, 1353, 1354, 1375, 1398, 1399, 1418, 1451, 1490, 1617, 1618, 1619, 1686, 1687, 1688, 1689, 1695, 1715, 1716, 1730, 1747, 1754, 1755, 1801, 1802, 1803, 1845, 1859, 1860, 1869, 1890, 1912, 1913, 1922, 1923, 1929]
col95=[652, 950, 1203, 1318, 1421, 1497, 1892]
col94=[576, 577, 653, 654, 699, 937, 1503, 1504]

#union of the above columns
col9=[541, 542, 543, 544, 575, 576, 577, 583, 584, 585, 586, 598, 599, 600, 607, 608, 609, 610, 641, 642, 643, 648, 649, 650, 651, 652, 653, 654, 655, 659, 660, 669, 670, 671, 672, 673, 674, 675, 676, 677, 691, 698, 699, 910, 912, 913, 917, 918, 919, 920, 921, 931, 934, 937, 944, 946, 950, 951, 964, 968, 970, 976, 977, 978, 979, 980, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1179, 1180, 1181, 1182, 1183, 1199, 1200, 1201, 1202, 1203, 1204, 1215, 1220, 1221, 1227, 1228, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1341, 1342, 1353, 1354, 1355, 1356, 1357, 1371, 1372, 1373, 1374, 1375, 1376, 1398, 1399, 1418, 1419, 1420, 1421, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1451, 1452, 1453, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1581, 1582, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1715, 1716, 1717, 1718, 1730, 1731, 1732, 1733, 1738, 1739, 1747, 1748, 1754, 1755, 1756, 1757, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1845, 1846, 1847, 1859, 1860, 1861, 1869, 1870, 1871, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1912, 1913, 1914, 1915, 1922, 1923, 1929]

notthere=[218,240]

#Columns with only one value or one value and NAs
rm=[214,207,213,840,847,1428,8,9,10,11,12,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,38,39,40,41,42,43,44,196,197,199,202,203,215,216,221,222,223,229,239,188,189,190,246,394,438,446,527,528,530]

#Columns with less than 20 nonzero values
small=[98, 106, 138, 191, 192, 193, 395, 396, 397, 398, 399, 411, 445, 459, 526, 529,493]


duplicates=[227,260,181,182,238,201,13,916,512]

#with column 130
neardup=[114]

selectColumns=[]
rmCol = rm+notthere+duplicates+small+neardup
for i in range(1,1935):
    if i not in rmCol:
        selectColumns.append(i)


cols = [str(n).zfill(4) for n in selectColumns]
strColName = ['VAR_' + strNum for strNum in cols] 

#Takes the non numerical columns and take the top four most common values (or less if 4 or less unique values)
change={'VAR_0001':['R','H'],'VAR_0005':['B','C','N'],'VAR_0200':['CHICAGO','HOUSTON','JACKSONVILLE','SAN ANTONIO'],'VAR_0226':['True'],'VAR_0230':['True'],'VAR_0232':['True'],'VAR_0236':['True'],'VAR_0237':['CA','TX','NC','GA','IL'],'VAR_0274':['CA','TX','NC','IL'],'VAR_0283':['S','H'],'VAR_0305':['S','H','P'],'VAR_0325':['H','S','P'],'VAR_0342':['FF','EE','FE','DD'],'VAR_0352':['U','O','R'],'VAR_0353':['U','O','R'],'VAR_0354':['U','O','R'],'VAR_0404':['CONTACT','PRESIDENT','AGENT','DIRECTOR'],'VAR_0466':['I'],'VAR_0467':['Discharged','Dismissed'],'VAR_0492':['REGISTERED NURSE','LICENSED PRACTICAL NURSE','PHARMACY TECHNICIAN','COSMETOLOGIST'],'VAR_1934':['IAPS','BRANCH','MOBILE','CSC']}

nrows=100
trainData = pd.read_csv("../input/train.csv", usecols=strColName,nrows=nrows)
label = pd.read_csv("../input/train.csv", usecols=['target'],nrows=nrows)

# extract the day of week, month, year, and monthyear from the timestaps
cols=[str(n).zfill(4) for n in dtCol]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'m']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: x.month))
  trainData[i+'my']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: (x.year-2010)*12+x.month))
  trainData[i+'d']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: x.weekday()))
  trainData[i+'y']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: x.year-2010))
  trainData.drop(i, axis=1, inplace=True)

#change 999999999 etc. to indicators variables
cols=[str(n).zfill(4) for n in col99]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'9']=trainData[i].map(ind99)
cols=[str(n).zfill(4) for n in col98]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'8']=trainData[i].map(ind98)
cols=[str(n).zfill(4) for n in col97]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'7']=trainData[i].map(ind97)
cols=[str(n).zfill(4) for n in col96]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'6']=trainData[i].map(ind96)
cols=[str(n).zfill(4) for n in col95]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'5']=trainData[i].map(ind95)
cols=[str(n).zfill(4) for n in col94]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'4']=trainData[i].map(ind94)

#replace 999999999 etc. with NAs
cols=[str(n).zfill(4) for n in col9]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i]=trainData[i].map(makesmall)

#change strings to indicators variables
for column in change:
  for value in change[column]:
    trainData[column+value]=trainData[column].map(lambda x: toIndicate(x,value))
  trainData.drop(column, axis=1, inplace=True)

trainData['target']=label['target']

trainData.to_csv('newtrain.csv',index=False)


#same thing for test.csv
trainData = pd.read_csv("../input/test.csv", usecols=strColName,nrows=nrows)
IDs = pd.read_csv("../input/test.csv", usecols=['ID'],nrows=nrows)


cols=[str(n).zfill(4) for n in dtCol]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'m']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: x.month))
  trainData[i+'my']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: (x.year-2010)*12+x.month))
  trainData[i+'d']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: x.weekday()))
  trainData[i+'y']=(pd.to_datetime(trainData[i],format ='%d%b%y:%H:%M:%S').map(lambda x: x.year-2010))
  trainData.drop(i, axis=1, inplace=True)

cols=[str(n).zfill(4) for n in col99]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'9']=trainData[i].map(ind99)
cols=[str(n).zfill(4) for n in col98]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'8']=trainData[i].map(ind98)
cols=[str(n).zfill(4) for n in col97]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'7']=trainData[i].map(ind97)
cols=[str(n).zfill(4) for n in col96]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'6']=trainData[i].map(ind96)
cols=[str(n).zfill(4) for n in col95]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'5']=trainData[i].map(ind95)
cols=[str(n).zfill(4) for n in col94]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i+'4']=trainData[i].map(ind94)

cols=[str(n).zfill(4) for n in col9]
for i in ['VAR_' + strNum for strNum in cols]:
  trainData[i]=trainData[i].map(makesmall)


for column in change:
  for value in change[column]:
    trainData[column+value]=trainData[column].map(lambda x: toIndicate(x,value))
  trainData.drop(column, axis=1, inplace=True)


trainData['ID']=IDs['ID']

trainData.to_csv('newtest.csv',index=False)
