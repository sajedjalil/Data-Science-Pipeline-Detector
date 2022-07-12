
import numpy as np
np.random.seed(71)

ITEMS = {'A':0.9, 'B':0.3}

def multilabel_fscore(y_true, y_pred):
    """
    ex1:
    y_true = [1, 2, 3]
    y_pred = [2, 3]
    return: 0.8
    
    ex2:
    y_true = ["None"]
    y_pred = [2, "None"]
    return: 0.666
    
    ex3:
    y_true = [4, 5, 6, 7]
    y_pred = [2, 4, 8, 9]
    return: 0.25
    
    """
    y_true, y_pred = set(y_true), set(y_pred)
    
    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    
    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)
    
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)

def get_y_true():
    y_true = []
    for k in ITEMS.keys():
        if ITEMS[k]>np.random.uniform():
            y_true.append(k)
    if len(y_true)==0:
        y_true = ['None']
    return y_true


print(np.mean([multilabel_fscore(get_y_true(),['A']) for i in range(9999)]))
print(np.mean([multilabel_fscore(get_y_true(),['B']) for i in range(9999)]))
print(np.mean([multilabel_fscore(get_y_true(),['A','B']) for i in range(9999)]))
print(np.mean([multilabel_fscore(get_y_true(),['None']) for i in range(9999)]))

