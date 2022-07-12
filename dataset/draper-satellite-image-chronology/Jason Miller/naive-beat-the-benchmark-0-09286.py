import pandas as pd
sub = pd.read_csv('../input/sample_submission.csv');
print("Don Dadda");
print("Don Data");
print("John DRAPER"); # pun intended
print("of this Internet paper") 
print("still killing cold capers");
sub['day'] = "4 5 1 3 2"
print("Fucking image classification. I'll do a real script after Santander's competition ends.");
sub.to_csv('naive_submit.csv', index=False)