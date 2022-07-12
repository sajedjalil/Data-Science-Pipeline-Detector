
import numpy as np

ITEMS = list('abcdefghij')

np.random.seed(71)

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

def calc_fscore(label, y_true_val, y_pred_val, th=0.5):
    
    y_true = [l for l,y in zip(label, y_true_val) if y==1]
    if len(y_true)==0:
        y_true = ["None"]
        
    y_pred = [l for l,y in zip(label, y_pred_val) if y>th]
    if len(y_pred)==0:
        y_pred = ["None"]
        
    return multilabel_fscore(y_true, y_pred)

def get_valid(n):
    label = np.random.choice(ITEMS, size=n, replace=False)
    y_true_val = np.random.randint(2,size=n)
    y_pred_val = np.random.uniform(size=n)
    return label, y_true_val, y_pred_val

def main(n, th=0.5):
    fscores = []
    for i in range(99999):
        label, y_true_val, y_pred_val = get_valid(n)
        score = calc_fscore(label, y_true_val, y_pred_val, th=th)
        fscores.append(score)
    print(np.mean(fscores))

#==============================================================================
# main
#==============================================================================

main(9, th=0.5) # 0.469798598246

main(3, th=0.5) # 0.417695176952

main(3, th=0.4) # 0.46478731454









