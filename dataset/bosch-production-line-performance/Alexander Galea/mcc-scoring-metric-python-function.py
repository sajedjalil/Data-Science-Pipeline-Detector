def score_prediction(y_true, y_pred):
    ''' Evaluate the Matthews correlation coefficient (MCC) between
        the predicted and observed response. TP = true positive,
        TN = true negative, FP = false positive, FN = false negative '''
    
    TP = (y_true & y_pred).sum()
    TN = (1 - (y_true & y_pred)).sum()
    FP = (y_pred[y_true == 0]).sum()
    FN = (1 - y_pred[y_true == 1]).sum()
    
    val = (TP * TN) - (FP * FN)
    val = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    return val