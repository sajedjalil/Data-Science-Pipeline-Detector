import pandas as pd

# Data loading
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print('#### GENERAL ####')

train_len = len(train)
test_len = len(test)

print('Train size: '+str(train_len))
print('Test size: '+str(test_len))

print('')

####################
#### Y/N fields ####
####################

print('#### FIELDS WITH 4 OR LESS VALUES ####')

for f in train.columns:
        if train[f].dtype=='object':
            listed_values = list(train[f].values)
            setted_values = set(listed_values)
            if(len(setted_values) <= 4):
                print(f+' has '+str(len(setted_values))+' value(s) ['+str(setted_values)+']')

print('')

##########################
#### PersonalField10* ####
##########################

print('#### EXPLORING PersonalField10* FIELDS ####')

# TRAIN

train_pf10A_m1_len = len(train[train['PersonalField10A'] == -1])
train_pf10B_m1_len = len(train[train['PersonalField10B'] == -1])
train_pf10A_m1_converted_len = len(train[(train['PersonalField10A'] == -1) & (train['QuoteConversion_Flag'] == 1)])
train_pf10B_m1_converted_len = len(train[(train['PersonalField10B'] == -1) & (train['QuoteConversion_Flag'] == 1)])

# PersonalField10* with -1 in train
print('Train PersonalField10A == -1: '+str(train_pf10A_m1_len)+' ('+str(round((train_pf10A_m1_len/train_len)*100, 2))+'% of train dataset)')
print('Train PersonalField10B == -1: '+str(train_pf10B_m1_len)+' ('+str(round((train_pf10B_m1_len/train_len)*100, 2))+'% of train dataset)')

# PersonalField10* with -1 in train and quote converted
print(  'Train PersonalField10A == -1 and quote converted: '+
        str(train_pf10A_m1_converted_len)+
        ' ('+str(round((train_pf10A_m1_converted_len/train_pf10A_m1_len)*100, 2))+'% of train -1)'
    )
print(  'Train PersonalField10B == -1 and quote converted: '+
        str(train_pf10B_m1_converted_len)+
        ' ('+str(round((train_pf10B_m1_converted_len/train_pf10B_m1_len)*100, 2))+'% of train -1)'
    )

# TEST

test_pf10A_m1_len = len(test[test['PersonalField10A'] == -1])
test_pf10B_m1_len = len(test[test['PersonalField10B'] == -1])

# PersonalField10* with -1 in test
print('Test PersonalField10A == -1: '+str(test_pf10A_m1_len)+' ('+str(round((test_pf10A_m1_len/test_len)*100, 2))+'% of test dataset)')
print('Test PersonalField10B == -1: '+str(test_pf10B_m1_len)+' ('+str(round((test_pf10B_m1_len/test_len)*100, 2))+'% of test dataset)')

print('')

#########################
#### PropertyField20 ####
#########################

print('#### EXPLORING PropertyField20 ####')

# TRAIN

train_prf20_n0_len = len(train[train['PropertyField20'] != 0])
train_prf20_n0_converted_len = len(train[(train['PropertyField20'] != 0) & (train['QuoteConversion_Flag'] == 1)])

print('Train PropertyField20 != 0: '+str(train_prf20_n0_len)+' ('+str(round((train_prf20_n0_len/train_len)*100, 2))+'% of train dataset)')
print(  'Train PropertyField20 != 0 and quote converted: '+
        str(train_prf20_n0_converted_len)+
        ' ('+str(round((train_prf20_n0_converted_len/train_prf20_n0_len)*100, 2))+'% of train != 0)'
    )

# TEST

test_prf20_n0_len = len(test[test['PropertyField20'] != 0])
print('Test PropertyField20 != 0: '+str(test_prf20_n0_len)+' ('+str(round((test_prf20_n0_len/test_len)*100, 2))+'% of test dataset)')

print('')

########################
#### PersonalField8 ####
########################

print('#### EXPLORING PersonalField8 ####')

# TRAIN

train_pef8_n1_len = len(train[train['PersonalField8'] != 1])
train_pef8_n1_converted_len = len(train[(train['PersonalField8'] != 1) & (train['QuoteConversion_Flag'] == 1)])

print('Train PersonalField8 != 1: '+str(train_pef8_n1_len)+' ('+str(round((train_pef8_n1_len/train_len)*100, 2))+'% of train dataset)')
print(  'Train PersonalField8 != 1 and quote converted: '+
        str(train_prf20_n0_converted_len)+
        ' ('+str(round((train_pef8_n1_converted_len/train_pef8_n1_len)*100, 2))+'% of train != 1)'
    )

# TEST

test_pef8_n1_len = len(test[test['PersonalField8'] != 1])
print('Test PersonalField8 != 1: '+str(test_pef8_n1_len)+' ('+str(round((test_pef8_n1_len/test_len)*100, 2))+'% of test dataset)')

print ('')

##############################
#### Field10 values count ####
##############################
