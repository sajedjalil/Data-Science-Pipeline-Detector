# For more detail, please see Faron's post and kernel


cimport cython
import numpy as np
cimport numpy as np
from sklearn.metrics import f1_score


@cython.boundscheck(False)
@cython.wraparound(False)
def f1_opt(np.ndarray[long, ndim=1] label, np.ndarray[double, ndim=1] preds):

    cdef int i, j, k, k1
    cdef double f1, score, f1None, pNone
    cdef long n = preds.shape[0]

    pNone = (1 - preds).prod()

    cdef np.ndarray[long, ndim= 1] idx = np.argsort(preds)[::-1]
    label = label[idx]
    preds = preds[idx]

    cdef np.ndarray[double, ndim = 2] DP_C = np.zeros((n + 2, n + 1), dtype=np.float)

    DP_C[0, 0] = 1.0
    for j in range(1, n):
        DP_C[0, j] = (1.0 - preds[j - 1]) * DP_C[0, j - 1]
    for i in range(1, n + 1):
        DP_C[i, i] = DP_C[i - 1, i - 1] * preds[i - 1]
        for j in range(i + 1, n + 1):
            DP_C[i, j] = preds[j - 1] * DP_C[i - 1, j - 1] + (1.0 - preds[j - 1]) * DP_C[i, j - 1]

    cdef np.ndarray[double, ndim = 1] DP_S = np.zeros((2 * n + 1,))
    cdef np.ndarray[double, ndim = 1] DP_SNone = np.zeros((2 * n + 1,))
    for i in range(1, 2 * n + 1):
        DP_S[i] = 1. / (1. * i)
        DP_SNone[i] = 1. / (1. * i + 1)

    score = -1
    cdef np.ndarray[double, ndim= 1] expectations = np.zeros(n + 1)
    cdef np.ndarray[double, ndim= 1] expectationsNone = np.zeros(n + 1)

    for k in range(n + 1)[::-1]:
        f1 = 0
        f1None = 0
        for k1 in range(n + 1):
            f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
            f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
        for i in range(1, 2 * k - 1):
            DP_S[i] = (1 - preds[k - 1]) * DP_S[i] + preds[k - 1] * DP_S[i + 1]
            DP_SNone[i] = (1 - preds[k - 1]) * DP_SNone[i] + preds[k - 1] * DP_SNone[i + 1]
        expectations[k] = f1
        expectationsNone[k] = f1None + 2 * pNone / (2 + k)

    if expectations.max() > expectationsNone.max():
        i = np.argsort(expectations)[n] - 1
        tp = label[:i + 1].sum()
        if tp > 0:
            precision = tp / (i + 1)
            recall = tp / label.sum()
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
    else:
        i = np.argsort(expectationsNone)[n] - 1
        tp = label[:i + 1].sum() if label.sum() != 0 else 1
        if tp > 0:
            precision = tp / (i + 2)
            recall = tp / max(label.sum(), 1)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0

    return f1

from multiprocessing import Pool


@cython.boundscheck(False)
@cython.wraparound(False)
def f1_group(np.ndarray[long, ndim=1] label, np.ndarray[double, ndim=1] preds, np.ndarray[long, ndim=1] group):

    cdef int i, start, end, j, s
    cdef double score = 0.
    cdef long m = group.shape[0]
    cdef long n = preds.shape[0]
    start = 0

    p = Pool()
    list_p = []
    for i in range(m):
        end = start + group[i]
        list_p.append(p.apply_async(f1_opt, (label[start:end], preds[start:end],)))
        start = end
    scores = [a.get() for a in list_p]
    p.close()
    p.join()
    return np.mean(scores)