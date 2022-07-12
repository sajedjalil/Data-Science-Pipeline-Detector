"""Collapses data into a time series for each combination.

Only runs on the first 10k lines of data here, which happen to all be
in week 3. Set 'NUM_ROWS' to None to run on all data (takes ~15hrs on 
my machine).

"""
# Imports
import pandas as pd
import csv
import hashlib
import numpy as np

# Constants
NUM_ROWS = 10000    # Set to None to process the entire data set
in_file = '../input/train.csv'
out_file = './time_series_test.csv'


def load_raw_data(train_file, num_train=None):
    """Loads and preps training and test data."""
    # Load files with pandas
    data = pd.read_csv(train_file, nrows=num_train)

    # Set some columns as integers
    for col in ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',
                'Producto_ID']:
        data[col] = data[col].astype('int')

    return data


def hash_it(x):
    """Hashes x."""
    return hashlib.md5(x.encode()).hexdigest()


def add_weekly_data(data_set, data_row, hash_key_index):
    """Adds new weekly data to data_set time-series dictionary."""
    semana = int(data_row['Semana'])
    data_set[hash_key_index][5+(semana-3)*3] = int(data_row['Venta_uni_hoy'])
    data_set[hash_key_index][5+(semana-3)*3+1] = \
        int(data_row['Dev_uni_proxima'])
    data_set[hash_key_index][5+(semana-3)*3+2] = \
        int(data_row['Demanda_uni_equil'])
    return


def add_new_entry(data_set, data_row, hash_key_index):
    """Creates a time-series entry in the dictionary data_set.

    key:  hash_key_index
    row format: [Agencia_ID, Canal_ID, Ruta_SAK, Cliente_ID, Producto_ID,
                 Wk3:Venta_uni_hoy, Wk3:Dev_uni_proxima, Wk3:Demanda_uni_equil,
                 Wk4:...]

    """
    # Add label data
    new_entry = [int(data_row[col]) for col in ['Agencia_ID', 'Canal_ID',
                                                'Ruta_SAK', 'Cliente_ID',
                                                'Producto_ID']]
    new_entry = new_entry + [np.nan] * 21

    # Add to data_set
    data_set[hash_key_index] = new_entry

    # Add week specific data
    add_weekly_data(data_set, data_row, hash_key_index)
    return


# Main script
if __name__ == "__main__":
    # Load original data
    print('Loading original data...')
    data = load_raw_data(in_file, num_train=NUM_ROWS)

    # Create an empty data holder
    data_set = {}

    # Loop through to add rows to data_set that contain time series data
    print('Collapsing data into time series...')
    ctr = 0
    for ii in range(len(data)):
        # Create label for hashing
        label = ''
        for col in ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',
                    'Producto_ID']:
            label += str(data.iloc[ii][col])

        # Compute hash_key_index
        hash_key_index = hash_it(label)

        # Check if already done
        if hash_key_index in data_set:
            add_weekly_data(data_set, data.iloc[ii, :], hash_key_index)
        else:
            add_new_entry(data_set, data.iloc[ii, :], hash_key_index)

        # Update counter and report
        ctr += 1
        if ctr % int(len(data) / 100.) == 0:
            print('{0} %'.format(round(100. * float(ctr) / len(data))))

    # Report number of rows
    time_series_rows = len(data_set.keys())
    print('Number of time series: {0}'.format(time_series_rows))

    # Write out dictionary to csv
    print('\nSaving results...')
    with open(out_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(['hash', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK',
                    'Cliente_ID', 'Producto_ID',
                    'Wk3:Venta_uni_hoy', 'Wk3:Dev_uni_proxima',
                    'Wk3:Demanda_uni_equil',
                    'Wk4:Venta_uni_hoy', 'Wk4:Dev_uni_proxima',
                    'Wk4:Demanda_uni_equil',
                    'Wk5:Venta_uni_hoy', 'Wk5:Dev_uni_proxima',
                    'Wk5:Demanda_uni_equil',
                    'Wk6:Venta_uni_hoy', 'Wk6:Dev_uni_proxima',
                    'Wk6:Demanda_uni_equil',
                    'Wk7:Venta_uni_hoy', 'Wk7:Dev_uni_proxima',
                    'Wk7:Demanda_uni_equil',
                    'Wk8:Venta_uni_hoy', 'Wk8:Dev_uni_proxima',
                    'Wk8:Demanda_uni_equil',
                    'Wk9:Venta_uni_hoy', 'Wk9:Dev_uni_proxima',
                    'Wk9:Demanda_uni_equil'
                    ])
        ctr = 0
        for key, value in list(data_set.items()):
            w.writerow([key] + value)
            ctr += 1
            if ctr % int(time_series_rows / 10.) == 0:
                print('{0} %'.format(round(100. * ctr / time_series_rows)))

    print('Done.')
