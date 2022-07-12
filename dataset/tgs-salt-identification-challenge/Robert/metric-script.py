import numpy as np


def calc_iou(actual,pred):
  intersection = np.count_nonzero(actual*pred)
  union = np.count_nonzero(actual) + np.count_nonzero(pred) - intersection
  iou_result = intersection/union if union!=0 else 0.
  return iou_result

def calc_ious(actuals,preds):
  ious_ = np.array([calc_iou(a,p) for a,p in zip(actuals,preds)])
  return ious_

def calc_precisions(thresholds,ious):
  thresholds = np.reshape(thresholds,(1,-1))
  ious = np.reshape(ious,(-1,1))
  ps = ious>thresholds
  mps = ps.mean(axis=1)
  return mps
  
def indiv_scores(masks,preds):
  masks[masks>0] = 1
  preds[preds>0] = 1
  ious = calc_ious(masks,preds)
  thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  
  precisions = calc_precisions(thresholds,ious)
  
  ###### Adjust score for empty masks
  emptyMasks = np.count_nonzero(masks.reshape((len(masks),-1)),axis=1)==0
  emptyPreds = np.count_nonzero(preds.reshape((len(preds),-1)),axis=1)==0
  adjust = (emptyMasks==emptyPreds).astype(np.float)
  precisions[emptyMasks] = adjust[emptyMasks]
  ###################
  return precisions

def calc_metric(masks,preds):
  return np.mean(indiv_scores(masks,preds))