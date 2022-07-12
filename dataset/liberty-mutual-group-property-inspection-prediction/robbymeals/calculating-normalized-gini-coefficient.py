'''
The original version of this script gave radically different results from the implementation in R.
I just ported the R version over directly.
'''

import numpy as np

def gini(solution, submission):                                                 
    df = sorted(zip(solution, submission), key=lambda x : (x[1], x[0]),  reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def normalized_gini(solution, submission):                                      
    normalized_gini = gini(solution, submission)/gini(solution, solution)       
    return normalized_gini                                                      


if __name__ == '__main__':                                                      
    import pandas as pd
    # test case 1
    desired_y = np.array(pd.read_csv("../input/train.csv")["Hazard"])
    predicted_y = np.copy(desired_y)
    predicted_y[:] = 1
    print(normalized_gini(desired_y, predicted_y))
    print(normalized_gini(desired_y[::-1], predicted_y))

    ## test case 2
    predicted_y = [i + (i%30) for i in range(100)] 
    desired_y = [i for i in range(100)]
    print(normalized_gini(predicted_y, desired_y))
    
    ## test case 3
    predicted_y = [i for i in range(100)] 
    desired_y = [i + (i%30) for i in range(100)]
    print(normalized_gini(predicted_y, desired_y))