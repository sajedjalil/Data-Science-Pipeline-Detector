
import numpy as np
import pandas as pd


def simulate_test(test):

    indices = np.random.choice(test.index.values, 2800000)
    test_debug = pd.concat([test, test.iloc[indices]], axis=0)
    test_debug['test_id'] = np.arange(len(test_debug))
    return test_debug.copy()


def main():

    test = pd.read_table('../input/test.tsv')
    print('test shape (1st stage): {}'.format(test.shape))

    test_debug = simulate_test(test)
    print('test shape (2nd stage): {}'.format(test_debug.shape))

    print(test_debug.head())
    print(test_debug.tail())

    test_debug.to_csv('test_debug.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
