#!/usr/bin/env python3
'''
Randomforest - plot: # of trees - accuracy

@Author: Hideki Ikeda
@Date 7/11/15
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

def main():
    # loading training data
    print('Loading training data')
    data = pd.read_csv('../input/train.csv')
    X_tr = data.values[:, 1:].astype(float)
    y_tr = data.values[:, 0]

    scores = list()
    scores_std = list()

    print('Start learning...')
    n_trees = [10, 15, 20, 25, 30, 40, 50, 70, 100, 150]
    for n_tree in n_trees:
        print(n_tree)
        recognizer = RandomForestClassifier(n_tree)
        score = cross_val_score(recognizer, X_tr, y_tr)
        scores.append(np.mean(score))
        scores_std.append(np.std(score))

    sc_array = np.array(scores)
    std_array = np.array(scores_std)
    print('Score: ', sc_array)
    print('Std  : ', std_array)

    #plt.figure(figsize=(4,3))
    plt.plot(n_trees, scores)
    plt.plot(n_trees, sc_array + std_array, 'b--')
    plt.plot(n_trees, sc_array - std_array, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('# of trees')
    plt.savefig('cv_trees.png')
    # plt.show()


if __name__ == '__main__':
    main()
