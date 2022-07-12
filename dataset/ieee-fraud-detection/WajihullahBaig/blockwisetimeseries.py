import numpy as np
# https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
class BlocwiseTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
            
if __name__ == '__main__':
    train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    btscv = BlocwiseTimeSeriesSplit(n_splits=3)        
    for train_idx, valid_idx in btscv.split(train):
        print("Train idx:", train_idx)
        print("Valid idx:", valid_idx)