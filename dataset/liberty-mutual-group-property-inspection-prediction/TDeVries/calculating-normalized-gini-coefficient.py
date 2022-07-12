'''
IMPORTANT EDIT!
This script calculates the Gini coefficient incorrectly, sorry to everyone who tried using it and got weird results.
robbymeals has made a better working version here: 
https://www.kaggle.com/rmealey/liberty-mutual-group-property-inspection-prediction/calculating-normalized-gini-coefficient/comment/83719#post83719
0X0FFF also made a working version and used it to train an XGBoost model:
https://www.kaggle.com/oxofff/liberty-mutual-group-property-inspection-prediction/gini-scorer-cv-gridsearch
Use their scripts instead of this thing.
'''

'''
ORIGINAL BROKEN SCRIPT BEGINS HERE
This script provides the functions required to calculate the normalized gini coefficient
Credit to https://planspacedotorg.wordpress.com/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/ for gini function
Apparently this method differs from the standard gini coefficient calculation method for some edge cases, but it gives a good idea of how it is calculated
'''

import numpy as np

def gini(list_of_values):
  sorted_list = sorted(list(list_of_values))
  height, area = 0, 0
  for value in sorted_list:
    height += value
    area += height - value / 2.
  fair_area = height * len(list_of_values) / 2
  return (fair_area - area) / fair_area
  
def normalized_gini(y_pred, y):
    normalized_gini = gini(y_pred)/gini(y)
    return normalized_gini
    

predicted_y = np.random.randint(100, size = 1000)
desired_y = np.random.randint(100, size = 1000)

print (normalized_gini(predicted_y, desired_y))