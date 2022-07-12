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
    
    
n = 100
noise = 30
power = 1.2

x0 = np.arange(n)
x = x0 - np.remainder(x0, noise)*1.5
y = x0**power

print(x)
print(y)
print(normalized_gini(solution=y, submission=x))