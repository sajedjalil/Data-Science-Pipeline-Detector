import time
import os.path
import numpy as np
import pandas as pd

COLUMN_TO_TYPE = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}

def prepare_data(directory, name, output_columns):
    start_time = time.time()
    file_path = os.path.join(directory, '{}.csv'.format(name))
    print('reading {}  '.format(file_path), end='')

    dtypes = {column: COLUMN_TO_TYPE[column] for column in output_columns}

    data = pd.read_csv(file_path, usecols =output_columns, dtype=dtypes, engine='c')
    print("{:6.4f} secs".format((time.time() - start_time)))

    for column in output_columns:
        output_file_name = '{}_{}.bin'.format(name, column)
        print('dumping {}  '.format(output_file_name), end='')
        start_time = time.time()
        mmap = np.memmap(output_file_name, dtype=COLUMN_TO_TYPE[column], mode='w+', shape=(data.shape[0]))
        mmap[:] = data[column].values
        del mmap
        print("{:6.4f} secs".format((time.time() - start_time)))


def main():
    directory = '../input'
    output_columns = ['flux', 'flux_err', 'detected']
    prepare_data(directory, 'test_set', output_columns)


if __name__ == '__main__':
    main()
