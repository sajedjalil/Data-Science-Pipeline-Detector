import pandas as pd
import numpy as np

CSData = pd.read_csv("../input/train.csv")
CSData_Target1 = CSData[CSData.TARGET != 0]

columnsUniqueValues = []
columnsUniqueValues_list = []
columnsUniqueValues_list_temp = []
columnsUniqueValuesCount = []
columnsUniqueValuesRecurrence = []
columnsUniqueValueLabels = []
columnsUniqueValuesRecurrence_list = []
columnsUniqueValuesRecurrence_list_target1 = []


# remove constant columns
remove = []
for col in CSData.columns:
    if CSData[col].std() == 0:
        remove.append(col)

CSData.drop(remove, axis=1, inplace=True)
CSData_Target1.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = CSData.columns
for i in range(len(c)-1):
    v = CSData[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,CSData[c[j]].values):
            remove.append(c[j])

CSData.drop(remove, axis=1, inplace=True)
CSData_Target1.drop(remove, axis=1, inplace=True)


columnsList = list(CSData.columns.values)
columnsList.remove("ID")

for columnName in columnsList:
    if columnName != 'ID':
        columnsUniqueValues.append(pd.unique(CSData[columnName]))
        columnsUniqueValuesCount.append(CSData[columnName].nunique())
        columnsUniqueValues_list_temp = list(pd.unique(CSData[columnName]))
        columnsUniqueValuesRecurrence = list(CSData[columnName].value_counts())
        for x in range(0, len(pd.unique(CSData[columnName]))):
            columnsUniqueValueLabels.append(columnName)
            columnsUniqueValues_list.append(columnsUniqueValues_list_temp[x])
            temp_df = CSData[CSData[columnName] == columnsUniqueValues_list_temp[x]]
            columnsUniqueValuesRecurrence_list.append(temp_df.shape[0])
            temp_df = CSData_Target1[CSData_Target1[columnName] == columnsUniqueValues_list_temp[x]]
            columnsUniqueValuesRecurrence_list_target1.append(temp_df.shape[0])

print(len(columnsUniqueValueLabels))
print(len(columnsUniqueValues_list))
print(len(columnsUniqueValuesRecurrence_list))

completeValuesDF = pd.DataFrame({'Column Name': columnsList, 'Unique Values': columnsUniqueValues, 'Count of Values': columnsUniqueValuesCount})
completeValuesDF.to_csv('uniqueValues.csv')


uniqueValueDetailsDF = pd.DataFrame({'Column Name': columnsUniqueValueLabels, 'Unique Values': columnsUniqueValues_list, 'Recurrence of Values': columnsUniqueValuesRecurrence_list, "Target 1 Count": columnsUniqueValuesRecurrence_list_target1})
uniqueValueDetailsDF.to_csv('uniqueValuesDetails.csv')