# kaggle submit functionality
import numpy as np
import pandas as pd
import tensorflow as tf

# functions definitions


def general_mean_scaler(local_array):
    if len(local_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_array, axis=1)
    mean_scaling = np.divide(local_array, 1 + mean_local_array)
    return mean_scaling, mean_local_array


def window_based_normalizer(local_window_array):
    if len(local_window_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_window_array, axis=1)
    window_based_normalized_array = np.add(local_window_array, -mean_local_array)
    return window_based_normalized_array, mean_local_array


def general_mean_rescaler(local_array, local_complete_array_unit_mean, local_forecast_horizon):
    if len(local_array) == 0:
        return "argument length 0"
    local_array = local_array.clip(0)
    local_complete_array_unit_mean = np.array([local_complete_array_unit_mean, ] * local_forecast_horizon).transpose()
    mean_rescaling = np.multiply(local_array, 1 + local_complete_array_unit_mean)
    return mean_rescaling


def window_based_denormalizer(local_window_array, local_last_window_mean, local_forecast_horizon):
    if len(local_window_array) == 0:
        return "argument length 0"
    local_last_window_mean = np.array([local_last_window_mean, ] * local_forecast_horizon).transpose()
    window_based_denormalized_array = np.add(local_window_array, local_last_window_mean)
    return window_based_denormalized_array


# main section
if __name__ == '__main__':
    try:
        # paths
        models_folder = '/kaggle/input/m5forecastmodel/'
        sales_folder = '/kaggle/input/m5-forecasting-accuracy/'
        groups_folder = '/kaggle/input/groups/'

        # constants
        nof_groups = 3
        forecast_horizon_days = 28
        moving_window_length = 56

        # load data
        raw_data_sales = pd.read_csv(''.join([sales_folder, 'sales_train_validation.csv']))
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        print('raw sales data accessed')

        # load clean data (divided in groups) and time_serie_group
        time_series_group = np.load(''.join([groups_folder, 'time_serie_group.npy']))
        print('time serie group division data loaded')
        preprocessed_unit_sales_g1 = np.load(''.join([groups_folder, 'group1.npy']))
        preprocessed_unit_sales_g2 = np.load(''.join([groups_folder, 'group2.npy']))
        preprocessed_unit_sales_g3 = np.load(''.join([groups_folder, 'group3.npy']))
        groups_list = [preprocessed_unit_sales_g1, preprocessed_unit_sales_g2, preprocessed_unit_sales_g3]
        print('preprocessed data by group aggregated loaded')

        # load models
        forecasters_list = []
        for group in range(nof_groups):
            model_group = tf.keras.models.load_model(''.join([models_folder, 'model_group_',
                                                              str(group),  '_forecast_.h5']))
            model_group.compile(optimizer='adam', loss='mse')
            forecasters_list.append(model_group)

        # preprocess data
        print('preprocess was externally made, but the reverse rescaling and denormalize needs some computations')
        nof_time_series = raw_unit_sales.shape[0]
        nof_selling_days = raw_unit_sales.shape[1]
        mean_unit_complete_time_serie = []
        scaled_unit_sales = np.zeros(shape=(nof_time_series, nof_selling_days))
        for time_serie in range(nof_time_series):
            scaled_time_serie = general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[0]
            mean_unit_complete_time_serie.append(general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[1])
            scaled_unit_sales[time_serie: time_serie + 1, :] = scaled_time_serie
        mean_unit_complete_time_serie = np.array(mean_unit_complete_time_serie)

        # make forecasts
        print('starting forecasts')
        nof_groups = len(groups_list)
        forecasts = np.zeros(shape=(nof_time_series * 2, forecast_horizon_days))
        for group in range(nof_groups):
            time_series_in_group = time_series_group[:, [0]][time_series_group[:, [1]] == group]
            group_data = groups_list[group]
            x_test = group_data[:, -forecast_horizon_days:]
            x_test = x_test.reshape(1, x_test.shape[1], x_test.shape[0])
            forecaster = forecasters_list[group]
            point_forecast_normalized = forecaster.predict(x_test)
            # inverse reshape
            point_forecast_reshaped = point_forecast_normalized.reshape((point_forecast_normalized.shape[2],
                                                                         point_forecast_normalized.shape[1]))
            # inverse transform (first moving_windows denormalizing and then general rescaling)
            time_serie_normalized_window_mean = np.mean(groups_list[group][:, -moving_window_length:], axis=1)
            denormalized_array = window_based_denormalizer(point_forecast_reshaped,
                                                           time_serie_normalized_window_mean,
                                                           forecast_horizon_days)
            group_time_serie_unit_sales_mean = []
            for time_serie in time_series_in_group:
                group_time_serie_unit_sales_mean.append(mean_unit_complete_time_serie[time_serie])
            point_forecast = general_mean_rescaler(denormalized_array,
                                                   np.array(group_time_serie_unit_sales_mean), forecast_horizon_days)
            # point_forecast = np.ceil(point_forecast[0, :, :])
            point_forecast = point_forecast.reshape(np.shape(point_forecast)[1], np.shape(point_forecast)[2])
            for time_serie_iterator in range(np.shape(point_forecast)[0]):
                forecasts[time_series_in_group[time_serie_iterator], :] = point_forecast[time_serie_iterator, :]
        print('time serie forecasts done')

        # submit results
        forecast_data_frame = np.genfromtxt(''.join([sales_folder, 'sample_submission.csv']), delimiter=',', dtype=None, encoding=None)
        forecast_data_frame[1:, 1:] = forecasts
        pd.DataFrame(forecast_data_frame).to_csv('submission.csv', index=False, header=None)
        print('submission done')

    except Exception as ee:
        print("Controlled error in main block___'___main_____'____")
        print(ee)
