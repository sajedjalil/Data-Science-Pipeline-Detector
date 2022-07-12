import time
import os.path
import numpy as np
import pandas as pd
import gc

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
        

    if 'object_id' in output_columns:
        to_remove = list(set(output_columns) - set(['object_id']))
        data.drop(to_remove, axis=1, inplace=True)
        gc.collect()
        object_ids = data['object_id'].values
        del data
        gc.collect()

        previous_object_id = -1
        object_id_range = []
        max_index = -1
        min_index = -1
        for index, object_id in enumerate(object_ids):
            if previous_object_id != object_id:
                if min_index != -1:
                    object_id_range.append([previous_object_id, min_index, max_index + 1])
                min_index = index

            previous_object_id = object_id
            max_index = index
            
        object_id_range.append([previous_object_id, min_index, max_index + 1])

        object_id_to_range = pd.DataFrame(data=object_id_range, columns=['object_id', 'start', 'end'])
        object_range_file_path = 'object_id_range.h5'
        print('dumping {}'.format(object_range_file_path))
        object_id_to_range.to_hdf(object_range_file_path, 'data', mode='w')


def main():
    directory = '../input'
    output_columns = ['object_id', 'mjd', 'passband']
    prepare_data(directory, 'test_set', output_columns)


if __name__ == '__main__':
    main()
