import gc
import os
import numpy as np 
import pandas as pd 
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns 
import itertools
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import keras

class GPSoftmax:
    def __init__(self):
        self.classes = 14
        self.class_names = [ 'class_6',
                             'class_15',
                             'class_16',
                             'class_42',
                             'class_52',
                             'class_53',
                             'class_62',
                             'class_64',
                             'class_65',
                             'class_67',
                             'class_88',
                             'class_90',
                             'class_92',
                             'class_95']


    def GrabPredictions(self, data):
        oof_preds = np.zeros((len(data), len(self.class_names)))
        oof_preds[:,0] = self.GP_class_6(data)
        oof_preds[:,1] = self.GP_class_15(data)
        oof_preds[:,2] = self.GP_class_16(data)
        oof_preds[:,3] = self.GP_class_42(data)
        oof_preds[:,4] = self.GP_class_52(data)
        oof_preds[:,5] = self.GP_class_53(data)
        oof_preds[:,6] = self.GP_class_62(data)
        oof_preds[:,7] = self.GP_class_64(data)
        oof_preds[:,8] = self.GP_class_65(data)
        oof_preds[:,9] = self.GP_class_67(data)
        oof_preds[:,10] = self.GP_class_88(data)
        oof_preds[:,11] = self.GP_class_90(data)
        oof_preds[:,12] = self.GP_class_92(data)
        oof_preds[:,13] = self.GP_class_95(data)
        oof_df = pd.DataFrame(np.exp(oof_preds), columns=self.class_names)
        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)
        return oof_df


    def GP_class_6(self,data):
        return (-1.965653 +
                0.100000*np.tanh(((((((((((data["flux_err_min"]) + ((((((data["0__kurtosis_y"]) < (data["flux_err_min"]))*1.)) * 2.0)))) / 2.0)) * 2.0)) * 2.0)) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (data["flux_err_min"]))) + (((data["flux_err_min"]) + (((((data["flux_err_min"]) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh(((data["flux_diff"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((((data["detected_flux_err_mean"]) * 2.0)) + (data["flux_err_min"]))) + (data["flux_err_median"]))) +
                0.100000*np.tanh(((data["flux_std"]) + (((data["4__skewness_x"]) + ((((((data["4__skewness_x"]) + (data["detected_flux_err_mean"]))) + (data["detected_flux_err_median"]))/2.0)))))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_mean"])), ((((((data["flux_err_median"]) * 2.0)) * 2.0)))))), ((data["flux_err_mean"])))) + (data["flux_err_mean"]))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((((data["flux_err_median"]) * 2.0))))) + (((data["flux_err_min"]) / 2.0)))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((((np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh(np.minimum(((np.where(data["4__kurtosis_x"] > -1, data["detected_flux_skew"], ((np.minimum(((data["detected_flux_skew"])), ((data["detected_flux_err_min"])))) * 2.0) ))), ((((data["detected_flux_err_min"]) + (data["detected_flux_skew"])))))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((((((data["flux_err_min"]) * 2.0)) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, data["5__kurtosis_x"], ((data["5__kurtosis_x"]) * 2.0) ))), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) * 2.0)) +
                0.100000*np.tanh((((data["detected_flux_err_min"]) + (((np.where((((5.20438098907470703)) + (data["detected_flux_err_median"]))>0, data["detected_flux_err_min"], data["detected_flux_err_min"] )) * 2.0)))/2.0)) +
                0.100000*np.tanh(((data["detected_flux_err_min"]) + (((np.minimum(((((((-1.0) - (data["detected_flux_min"]))) - (data["distmod"])))), ((data["detected_flux_min"])))) - (data["distmod"]))))) +
                0.100000*np.tanh(np.where(data["flux_err_min"]>0, data["5__kurtosis_x"], np.where(data["mwebv"] > -1, np.minimum(((((data["detected_flux_min"]) * 2.0))), ((((data["flux_err_min"]) * 2.0)))), data["flux_err_min"] ) )) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["detected_flux_min"])), ((data["flux_err_min"]))))), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_skew"])), ((((data["flux_err_min"]) * (((data["detected_flux_min"]) * (data["flux_err_min"])))))))) + (((data["flux_err_min"]) + (data["flux_err_min"]))))) +
                0.100000*np.tanh(((((data["4__skewness_x"]) + (np.minimum(((data["flux_err_min"])), ((np.minimum(((data["flux_err_min"])), ((data["4__skewness_x"]))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["4__skewness_x"])), ((((((((data["flux_err_min"]) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["flux_err_min"])), ((data["detected_flux_err_std"]))))))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_err_min"]<0, data["flux_err_min"], ((data["flux_err_min"]) * 2.0) )) * 2.0)) + (np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["flux_err_min"]))) * 2.0)) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) * (data["detected_flux_max"]))) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz"] > -1, data["detected_flux_err_max"], data["5__kurtosis_x"] ) > -1, np.where(data["distmod"] > -1, data["distmod"], data["5__skewness_x"] ), data["5__kurtosis_x"] )) +
                0.100000*np.tanh(((data["flux_max"]) * (np.minimum(((np.minimum(((((data["detected_flux_min"]) + (data["detected_flux_skew"])))), ((data["flux_max"]))))), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((data["detected_flux_min"]) - (data["distmod"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -3.0, np.where(data["hostgal_photoz"] > -1, data["distmod"], (((-1.0*((-3.0)))) + (data["detected_flux_min"])) ) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((data["flux_err_min"])))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_skew"])), ((data["5__skewness_x"])))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((data["flux_err_min"]) * 2.0))), ((((((((np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, np.where(np.where(data["distmod"] > -1, data["distmod"], 3.141593 ) > -1, -3.0, data["distmod"] ), data["5__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(-2.0 > -1, ((-2.0) - (data["distmod"])), ((-2.0) - (data["distmod"])) )) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (((((np.minimum(((data["flux_d0_pb4"])), ((data["detected_flux_max"])))) + (np.minimum(((data["flux_max"])), ((data["flux_err_min"])))))) * (data["flux_max"]))))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) * (((np.minimum(((((data["flux_d0_pb4"]) * (data["detected_flux_max"])))), ((data["detected_flux_skew"])))) * (data["detected_flux_max"]))))) - (data["detected_flux_max"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_flux_skew"])))))) * (((data["detected_flux_min"]) + (data["detected_flux_skew"]))))) +
                0.100000*np.tanh(np.where((((data["distmod"]) + (data["distmod"]))/2.0) > -1, -1.0, np.where(((data["distmod"]) / 2.0) > -1, -1.0, (-1.0*((data["distmod"]))) ) )) +
                0.100000*np.tanh(((((np.minimum(((((((-2.0) * 2.0)) - (data["distmod"])))), ((-2.0)))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["5__skewness_x"])), ((((data["flux_err_min"]) + ((((((-1.0) * 2.0)) + (data["detected_flux_skew"]))/2.0)))))))))) +
                0.100000*np.tanh(np.where(data["flux_d0_pb4"]<0, (((data["flux_by_flux_ratio_sq_skew"]) + (((data["detected_flux_max"]) * (data["flux_d0_pb4"]))))/2.0), data["detected_flux_max"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, np.where(-3.0 > -1, -3.0, -3.0 ), ((data["4__skewness_x"]) - (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(((((((((-3.0) - (data["distmod"]))) * 2.0)) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((np.where(((data["distmod"]) - (-2.0))<0, -2.0, ((((data["distmod"]) - (-2.0))) - (data["distmod"])) ))))) +
                0.100000*np.tanh((-1.0*((((data["distmod"]) + (((((((data["detected_mjd_diff"]) + (data["distmod"]))) + (3.141593))) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((((-1.0) - (data["distmod"]))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )) - (3.141593)), ((data["detected_flux_min"]) * (data["flux_max"])) )) +
                0.100000*np.tanh(np.where(3.141593 > -1, (-1.0*((np.where(((data["distmod"]) / 2.0) > -1, 2.0, data["distmod"] )))), ((-1.0) - (data["distmod"])) )) +
                0.100000*np.tanh(((((((data["flux_d0_pb4"]) * (data["flux_max"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.minimum(((((-2.0) - (data["distmod"])))), ((-1.0)))) - (data["distmod"]))) +
                0.100000*np.tanh(((data["detected_flux_max"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (((data["flux_max"]) * (((data["detected_flux_min"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))))))) +
                0.100000*np.tanh(((np.minimum((((((-1.0*(((((2.0) + (data["distmod"]))/2.0))))) * 2.0))), ((((data["detected_flux_err_std"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh((((((((data["4__skewness_y"]) * (((((data["0__skewness_x"]) * 2.0)) * 2.0)))) + (((data["0__skewness_x"]) - (data["detected_mjd_diff"]))))/2.0)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(((data["mwebv"]) - (data["flux_err_min"])) > -1, ((-3.0) * (((data["mwebv"]) - (data["flux_err_min"])))), data["flux_skew"] )) +
                0.100000*np.tanh(np.where((((data["detected_flux_err_std"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)>0, data["5__fft_coefficient__coeff_0__attr__abs__y"], np.tanh((data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) +
                0.100000*np.tanh(((((((((np.minimum(((-2.0)), ((-2.0)))) - (data["distmod"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_min"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["4__skewness_y"]))) + ((((3.0)) * (((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (data["detected_flux_min"]))))))))) +
                0.100000*np.tanh(((((((data["detected_flux_median"]) * (data["detected_flux_max"]))) + (data["4__skewness_y"]))) + (((((data["flux_min"]) * (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((data["5__skewness_y"]) - (data["detected_mjd_diff"])), ((data["detected_mjd_diff"]) - (data["detected_flux_max"])) )) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) + (np.minimum(((data["5__kurtosis_x"])), (((((data["4__skewness_x"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))/2.0))))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_err_skew"])), ((((((((data["detected_flux_err_skew"]) + (data["3__skewness_y"]))) + (data["detected_flux_err_skew"]))) + (((data["3__skewness_y"]) + (data["detected_flux_err_skew"])))))))) +
                0.100000*np.tanh((((((-1.0*((np.where(np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["flux_err_min"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ) > -1, data["distmod"], data["detected_flux_err_std"] ))))) - (data["detected_flux_err_std"]))) * 2.0)) +
                0.100000*np.tanh(((((((((np.where(data["detected_mjd_diff"] > -1, data["3__skewness_y"], data["detected_mjd_diff"] )) - (data["detected_mjd_diff"]))) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((data["flux_err_min"])), ((data["detected_flux_err_std"]))))), ((data["flux_err_min"])))) + (data["flux_err_min"]))) * 2.0)) +
                0.100000*np.tanh((((data["flux_d0_pb4"]) + (data["flux_d0_pb4"]))/2.0)) +
                0.100000*np.tanh(((3.0) - (np.where(((data["distmod"]) / 2.0) > -1, (14.57934379577636719), data["hostgal_photoz"] )))) +
                0.100000*np.tanh(((data["flux_min"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((np.where(data["mwebv"] > -1, ((data["flux_err_min"]) - (data["mwebv"])), data["2__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (np.where((6.0) > -1, np.where(data["hostgal_photoz"] > -1, (6.0), data["hostgal_photoz"] ), data["1__fft_coefficient__coeff_0__attr__abs__y"] )))) +
                0.100000*np.tanh(((np.minimum(((data["1__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, data["distmod"], np.maximum(((((data["detected_flux_err_std"]) / 2.0))), ((np.maximum(((data["detected_flux_err_std"])), ((data["distmod"])))))) )) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["5__skewness_y"], data["detected_mjd_diff"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_max"]>0, data["detected_flux_err_skew"], np.where(data["flux_max"]>0, np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, data["detected_flux_err_skew"], ((data["detected_flux_err_skew"]) * 2.0) ), data["detected_flux_err_max"] ) )) +
                0.100000*np.tanh((-1.0*((((2.718282) + (((data["distmod"]) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) - (data["detected_flux_max"]))) + (((data["flux_min"]) + (np.minimum(((data["1__skewness_x"])), ((data["1__skewness_x"])))))))) +
                0.100000*np.tanh((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((np.where(data["detected_flux_max"] > -1, data["detected_flux_min"], ((data["detected_flux_err_median"]) + (0.367879)) )) + (data["2__skewness_x"])), data["detected_flux_err_max"] )) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_err_max"] )) + (((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)) + (data["2__kurtosis_x"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) * (((np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["flux_d1_pb5"], np.where(data["2__skewness_y"]<0, data["flux_d0_pb0"], data["flux_std"] ) )) - (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((data["detected_flux_err_mean"]) * 2.0)) + (data["detected_flux_err_max"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) - (np.where(np.tanh((data["1__fft_coefficient__coeff_1__attr__abs__y"])) > -1, data["mwebv"], data["mwebv"] )))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((data["4__skewness_y"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["1__kurtosis_x"], data["flux_min"] )))/2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]>0, (((data["3__skewness_y"]) < (data["5__skewness_y"]))*1.), (((data["flux_median"]) + (((data["detected_flux_w_mean"]) + (((data["1__kurtosis_x"]) * 2.0)))))/2.0) )) +
                0.100000*np.tanh(np.maximum(((np.where(data["hostgal_photoz"] > -1, -2.0, np.maximum(((((data["flux_min"]) + (data["flux_err_min"])))), ((-2.0))) ))), ((-2.0)))) +
                0.100000*np.tanh(((((data["5__skewness_y"]) + (data["flux_err_min"]))) - (np.where(((data["detected_mjd_diff"]) + (((data["5__skewness_y"]) / 2.0))) > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d1_pb0"])), ((((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__kurtosis_x"], np.minimum(((data["detected_flux_err_skew"])), ((data["1__kurtosis_x"]))) )) * 2.0))))) +
                0.100000*np.tanh(((np.maximum(((((np.tanh((data["flux_d1_pb0"]))) * (data["mjd_diff"])))), ((data["detected_flux_err_skew"])))) * (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]<0, data["flux_d1_pb1"], np.tanh((((np.maximum(((data["detected_flux_err_max"])), ((((data["4__skewness_x"]) - (data["flux_err_skew"])))))) + (data["4__skewness_x"])))) )) +
                0.100000*np.tanh(((1.0) * (data["detected_flux_err_median"]))) +
                0.100000*np.tanh(((np.minimum(((((np.tanh((data["5__skewness_y"]))) * (data["2__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_by_flux_ratio_sq_sum"])))) * (np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_ratio_sq_skew"], data["2__skewness_y"] )))) +
                0.100000*np.tanh(((((((-2.0) - (data["distmod"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((-1.0*((((data["distmod"]) + (np.maximum(((2.0)), ((((data["distmod"]) + (data["distmod"]))))))))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]>0, ((data["1__skewness_x"]) * 2.0), np.where(((data["flux_dif2"]) * 2.0)<0, ((data["detected_flux_max"]) * (data["detected_flux_dif3"])), data["flux_err_skew"] ) )) +
                0.100000*np.tanh((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__kurtosis_x"]))/2.0)) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) * (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) * (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["5__skewness_y"], ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) ) )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((((data["3__kurtosis_y"]) + (data["flux_err_min"]))/2.0)) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((data["flux_d0_pb4"]) * (data["3__skewness_x"]))) +
                0.100000*np.tanh(np.where(((-2.0) - (data["distmod"]))<0, np.where(-2.0<0, ((-2.0) - (data["distmod"])), data["detected_flux_median"] ), 2.718282 )) +
                0.100000*np.tanh(((data["flux_median"]) + (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_min"]))) + (data["flux_min"]))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) > (data["0__kurtosis_x"]))*1.)) + (np.minimum(((data["flux_err_min"])), ((data["flux_err_min"])))))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"]<0, data["detected_flux_skew"], np.minimum(((data["flux_err_min"])), ((data["flux_err_min"]))) )) +
                0.100000*np.tanh(((data["0__kurtosis_x"]) + ((((((data["0__kurtosis_x"]) + (((data["3__skewness_y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["detected_mjd_diff"]) / 2.0)))) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_sum"]) * (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) * (data["detected_flux_by_flux_ratio_sq_sum"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) - (data["mwebv"]))) - (data["mwebv"]))) - (data["mwebv"]))) +
                0.100000*np.tanh(np.where(data["5__skewness_x"]<0, data["flux_err_skew"], ((data["1__kurtosis_x"]) * ((((data["detected_flux_ratio_sq_sum"]) > (np.maximum(((data["flux_mean"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))))*1.))) )) +
                0.100000*np.tanh(((np.maximum(((data["flux_median"])), ((data["flux_median"])))) * (((data["5__skewness_x"]) / 2.0)))) +
                0.100000*np.tanh(np.where(np.tanh((2.718282)) > -1, np.where(data["detected_flux_by_flux_ratio_sq_sum"]<0, np.where(data["flux_err_skew"] > -1, data["detected_flux_err_skew"], data["detected_flux_skew"] ), data["5__fft_coefficient__coeff_0__attr__abs__x"] ), data["mjd_diff"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_skew"]) * (np.tanh((data["detected_flux_err_skew"]))))) * (((data["4__skewness_x"]) * (data["detected_flux_err_skew"]))))) +
                0.100000*np.tanh(((((np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) * 2.0)) * (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["mwebv"]<0, data["flux_d1_pb2"], ((data["flux_err_min"]) - (data["mwebv"])) )) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]<0, ((((data["flux_err_skew"]) + (data["flux_err_skew"]))) + (data["detected_flux_median"])), ((data["detected_flux_median"]) + ((-1.0*((data["detected_flux_median"]))))) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["flux_d1_pb0"]>0, data["detected_flux_err_median"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["flux_err_min"], data["flux_err_min"] ) ), data["flux_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"] > -1, ((data["4__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0), np.tanh((data["detected_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"] > -1, data["flux_err_min"], data["flux_err_min"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))

    def GP_class_15(self,data):
        return (-1.349153 +
                0.100000*np.tanh(((data["flux_d1_pb0"]) + (np.minimum(((data["0__skewness_x"])), ((((((data["0__skewness_x"]) - (data["mjd_size"]))) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) + (np.minimum(((((data["flux_d0_pb1"]) + (((data["flux_d1_pb0"]) + (data["flux_d1_pb0"])))))), ((data["5__kurtosis_y"])))))) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) + (data["distmod"]))) + (((((np.minimum(((data["0__skewness_x"])), ((data["flux_d0_pb0"])))) - (data["detected_mjd_size"]))) * 2.0)))) +
                0.100000*np.tanh(np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["distmod"]) + (((data["detected_flux_min"]) * 2.0))))))), ((data["flux_d0_pb0"])))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_d1_pb0"])), ((data["flux_d0_pb0"]))))), ((((np.minimum(((data["0__skewness_x"])), ((data["detected_flux_min"])))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["0__skewness_x"])), ((((data["flux_d1_pb0"]) * 2.0))))) * 2.0))), ((((data["flux_d1_pb0"]) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb1"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), (((-1.0*((data["5__fft_coefficient__coeff_0__attr__abs__y"])))))))))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["0__skewness_x"]) + (data["detected_flux_min"]))))) + (data["flux_err_std"]))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((np.minimum(((data["0__skewness_x"])), ((data["flux_d0_pb0"])))) + (((data["flux_d0_pb0"]) + (data["distmod"]))))))) +
                0.100000*np.tanh(np.minimum(((((data["flux_d0_pb0"]) + (data["5__kurtosis_y"])))), ((data["flux_d1_pb0"])))) +
                0.100000*np.tanh(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["1__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["1__skewness_x"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["flux_d1_pb1"]) - (1.0))) - (((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (((((data["flux_d1_pb0"]) - (data["ddf"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((((((data["2__kurtosis_x"]) + (np.tanh((data["0__skewness_x"]))))) + (data["detected_flux_min"]))) + (np.where(data["distmod"] > -1, data["ddf"], data["detected_flux_min"] )))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (((((data["distmod"]) + (((data["0__skewness_x"]) + (data["detected_flux_min"]))))) + (data["distmod"]))))) +
                0.100000*np.tanh((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["flux_d0_pb1"]))/2.0)) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((data["1__skewness_x"]) + (((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_flux_min"]) * 2.0)))) + (np.tanh((((data["distmod"]) + (data["flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((data["hostgal_photoz_err"])))) + (((((data["detected_flux_min"]) + (((data["distmod"]) + (data["mjd_diff"]))))) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((data["flux_ratio_sq_skew"]) + (np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"]))))))), ((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))) + (data["flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((np.minimum(((data["flux_ratio_sq_skew"])), ((data["distmod"])))) + (data["flux_ratio_sq_skew"]))) + (data["1__skewness_x"]))) + (np.minimum(((data["flux_err_median"])), ((data["flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) - (data["4__kurtosis_x"]))) + (np.minimum(((data["flux_d1_pb1"])), ((np.minimum(((data["flux_d0_pb0"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))))))))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) - (np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]<0, np.where(data["detected_mjd_size"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d0_pb0"] ), data["detected_mjd_size"] )))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["distmod"]))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + ((((((data["flux_ratio_sq_skew"]) + (np.minimum(((data["flux_err_std"])), ((data["distmod"])))))/2.0)) + (data["flux_err_median"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (np.minimum(((np.minimum(((((data["distmod"]) - (0.0)))), ((data["flux_d1_pb1"]))))), ((data["distmod"])))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["flux_ratio_sq_skew"]))))), ((data["1__skewness_x"]))))), ((data["mjd_diff"]))))), ((data["flux_ratio_sq_skew"])))) +
                0.100000*np.tanh(((data["distmod"]) + (((((data["mjd_diff"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["flux_d1_pb1"]) + (((((data["distmod"]) + (data["detected_flux_min"]))) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (np.minimum(((data["detected_flux_min"])), ((data["distmod"])))))))))) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["distmod"]))) + (data["1__skewness_x"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((data["detected_flux_by_flux_ratio_sq_skew"]) + (np.minimum(((((data["detected_flux_err_min"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) + (((data["distmod"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((((((data["4__skewness_x"]) - (data["ddf"]))) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_mean"]))))) - (data["ddf"]))) +
                0.100000*np.tanh(((data["flux_d0_pb1"]) - (np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["flux_d0_pb1"]))>0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["1__skewness_x"] )))) +
                0.100000*np.tanh(((((((data["detected_flux_ratio_sq_skew"]) - (data["ddf"]))) + (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_d1_pb1"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["ddf"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (data["detected_flux_ratio_sq_skew"]))) + (((data["distmod"]) + (data["4__kurtosis_x"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["flux_d0_pb1"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb1"]) + (np.minimum(((data["flux_err_mean"])), ((data["distmod"])))))) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) + (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_by_flux_ratio_sq_skew"])))) + (data["4__skewness_x"])))))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) - (data["flux_d0_pb0"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]>0, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"])), data["3__kurtosis_x"] ), data["3__skewness_x"] )) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((data["ddf"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["detected_flux_ratio_sq_skew"]) + (((data["distmod"]) + (((data["mjd_diff"]) + (((data["flux_d0_pb0"]) + (data["mjd_diff"]))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.tanh((data["4__kurtosis_x"]))) * (((((data["detected_flux_min"]) * (data["4__kurtosis_x"]))) + (((data["detected_flux_min"]) * (data["4__kurtosis_x"]))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["ddf"]))) + (((((data["flux_ratio_sq_skew"]) - (((data["ddf"]) - (data["detected_flux_min"]))))) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((np.minimum(((data["hostgal_photoz_err"])), ((data["distmod"])))) + (((data["hostgal_photoz_err"]) + (data["detected_flux_min"]))))))))) +
                0.100000*np.tanh(np.where(np.where(data["detected_flux_err_min"]<0, data["detected_flux_ratio_sq_skew"], data["5__kurtosis_x"] )<0, ((data["detected_flux_err_min"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["detected_flux_ratio_sq_sum"])), ((data["3__kurtosis_x"]))))), ((((data["distmod"]) + (data["flux_ratio_sq_skew"]))))))), ((data["detected_flux_min"])))) +
                0.100000*np.tanh(np.where((((data["flux_d0_pb0"]) < (data["flux_ratio_sq_sum"]))*1.)>0, data["detected_flux_err_min"], 0.367879 )) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * 2.0)) + (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((data["detected_mean"]) < (((data["flux_d1_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))*1.)) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_mean"]>0, np.where(data["hostgal_photoz_err"]>0, data["distmod"], np.where(data["detected_mean"]>0, data["distmod"], data["distmod"] ) ), data["flux_ratio_sq_sum"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_d0_pb1"], (((-1.0*((data["4__fft_coefficient__coeff_1__attr__abs__y"])))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["flux_ratio_sq_sum"]))) + (data["flux_d1_pb5"]))))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((-1.0*((data["3__fft_coefficient__coeff_0__attr__abs__y"])))) > (data["detected_flux_err_min"]))*1.)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["flux_d1_pb3"]) * (np.where(np.minimum(((data["ddf"])), ((data["5__skewness_x"]))) > -1, data["2__kurtosis_x"], np.where(data["5__skewness_x"] > -1, data["4__kurtosis_x"], data["5__skewness_x"] ) )))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, ((((data["flux_d0_pb0"]) - (data["detected_flux_std"]))) * 2.0), ((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_ratio_sq_sum"]) * 2.0))) )) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_diff"]>0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["4__kurtosis_x"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["4__kurtosis_x"] ) ), data["1__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((np.where(data["detected_flux_std"]>0, data["detected_mjd_diff"], ((data["2__kurtosis_x"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"])) )) + (np.where(data["2__kurtosis_x"]>0, data["2__kurtosis_x"], data["flux_d0_pb5"] )))) +
                0.100000*np.tanh(((np.where(np.minimum(((data["detected_flux_err_min"])), ((data["detected_flux_ratio_sq_sum"])))>0, data["4__kurtosis_x"], data["distmod"] )) * (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["mjd_diff"], np.maximum(((data["flux_dif2"])), ((((data["2__kurtosis_x"]) - (np.minimum(((data["flux_dif2"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))))) )) +
                0.100000*np.tanh((((data["distmod"]) + (np.where(data["detected_flux_err_min"]>0, data["5__skewness_x"], ((np.minimum(((data["5__skewness_x"])), ((data["3__kurtosis_x"])))) + (data["flux_d0_pb5"])) )))/2.0)) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_median"]))*1.)) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) + (data["flux_d1_pb2"]))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_flux_std"], ((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_flux_std"])) )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], data["flux_ratio_sq_sum"] ), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_min"]))) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh((((((data["hostgal_photoz_err"]) * 2.0)) + (np.where((((data["flux_ratio_sq_skew"]) + ((((data["hostgal_photoz_err"]) + (data["distmod"]))/2.0)))/2.0)>0, data["flux_ratio_sq_skew"], data["hostgal_photoz_err"] )))/2.0)) +
                0.100000*np.tanh(((np.where(np.where(data["4__kurtosis_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )<0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["detected_mjd_diff"] )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_d0_pb1"]) + (((data["flux_ratio_sq_skew"]) - (data["mwebv"]))))) - (data["ddf"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where((((data["4__kurtosis_x"]) + (data["3__kurtosis_x"]))/2.0)<0, ((data["4__kurtosis_x"]) - (data["detected_flux_skew"])), data["flux_d1_pb5"] )) * (data["detected_flux_skew"]))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (((np.minimum(((data["distmod"])), ((data["distmod"])))) * 2.0)))))) +
                0.100000*np.tanh(((((((np.where((-1.0*((data["detected_mjd_diff"]))) > -1, data["detected_mjd_diff"], (-1.0*((data["detected_mjd_diff"]))) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]>0, data["flux_d1_pb5"], ((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_max"]))) * 2.0)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.minimum((((((((data["flux_median"]) * 2.0)) + (data["flux_median"]))/2.0))), ((data["mjd_size"])))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, np.maximum(((((((data["flux_d0_pb1"]) * 2.0)) + (np.maximum(((data["flux_d0_pb1"])), ((data["2__kurtosis_x"]))))))), ((data["detected_flux_ratio_sq_sum"]))), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(((data["2__kurtosis_x"]) + (np.maximum(((((data["1__kurtosis_y"]) + (data["2__kurtosis_x"])))), ((((data["2__kurtosis_x"]) + (data["2__kurtosis_x"])))))))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["detected_flux_ratio_sq_sum"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) + (data["detected_flux_err_min"]))) + ((((data["detected_flux_max"]) < (data["detected_flux_min"]))*1.)))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_min"]<0, ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_dif2"]) * 2.0) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["flux_err_mean"]) - (np.where(data["flux_d1_pb1"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["4__kurtosis_y"] )))) + (data["5__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["detected_flux_ratio_sq_sum"]))) + (data["detected_mjd_diff"]))) + (((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((data["flux_ratio_sq_sum"])), ((data["3__kurtosis_x"])))))))) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb5"]<0, np.where(data["1__kurtosis_y"]<0, data["1__kurtosis_y"], data["distmod"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]>0, data["flux_ratio_sq_sum"], np.where(data["detected_flux_err_min"]>0, data["flux_ratio_sq_sum"], ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_ratio_sq_sum"]))) * 2.0) ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_median"] > -1, data["detected_mjd_diff"], data["detected_flux_std"] )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]>0, data["2__kurtosis_x"], (((data["flux_by_flux_ratio_sq_sum"]) > (data["4__fft_coefficient__coeff_0__attr__abs__y"]))*1.) )) +
                0.100000*np.tanh(((((((np.where(data["flux_err_median"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], np.maximum(((((data["flux_median"]) * 2.0))), ((data["detected_flux_err_min"]))) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["mjd_diff"], np.where(data["mjd_diff"]>0, data["distmod"], data["distmod"] ) )) +
                0.100000*np.tanh((((((data["flux_median"]) < (np.where(data["3__skewness_x"]<0, data["detected_mjd_diff"], data["detected_flux_ratio_sq_sum"] )))*1.)) * (np.where(data["1__kurtosis_y"]>0, data["flux_median"], data["1__kurtosis_y"] )))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * (data["detected_flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((np.maximum(((data["2__kurtosis_x"])), ((data["flux_d1_pb3"])))) + (data["3__skewness_x"]))) * (((data["2__kurtosis_x"]) * (data["flux_d1_pb2"]))))) +
                0.100000*np.tanh(((data["distmod"]) + ((((((data["distmod"]) + (data["distmod"]))) + (data["hostgal_photoz_err"]))/2.0)))) +
                0.100000*np.tanh(np.where(data["detected_flux_median"]>0, data["4__kurtosis_x"], np.where(data["0__kurtosis_x"]>0, data["detected_flux_ratio_sq_sum"], data["4__kurtosis_x"] ) )) +
                0.100000*np.tanh((((((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) > (np.where(data["flux_d1_pb0"]>0, data["detected_flux_std"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )))*1.)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) > (data["hostgal_photoz"]))*1.)) +
                0.100000*np.tanh(np.where(data["flux_diff"]>0, data["detected_flux_err_min"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["4__skewness_x"] )) +
                0.100000*np.tanh((((data["flux_d0_pb1"]) > (data["detected_flux_dif3"]))*1.)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, np.where(((data["flux_err_skew"]) * 2.0)<0, data["2__kurtosis_x"], data["flux_median"] ), data["flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) * (data["detected_flux_err_skew"]))) * 2.0)) +
                0.100000*np.tanh(((np.maximum(((data["detected_flux_median"])), ((data["1__skewness_y"])))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((data["flux_d0_pb2"]) > (np.where(np.maximum(((np.minimum(((data["flux_d0_pb2"])), ((data["detected_flux_dif3"]))))), ((data["flux_d0_pb2"]))) > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb2"] )))*1.)) +
                0.100000*np.tanh(np.where(data["4__skewness_x"] > -1, np.where(data["4__skewness_x"]<0, data["hostgal_photoz_err"], data["1__skewness_y"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((np.where(np.where(data["detected_mjd_diff"]>0, data["flux_d0_pb3"], data["flux_d1_pb0"] )>0, data["distmod"], data["flux_d0_pb1"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(((data["detected_flux_err_median"]) * 2.0)<0, data["flux_d1_pb0"], np.where(data["flux_d1_pb0"]>0, data["distmod"], data["flux_d1_pb5"] ) )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_median"]) * 2.0)) + (((data["flux_median"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((np.maximum(((data["detected_mjd_diff"])), ((data["flux_err_skew"])))) + (((data["flux_d1_pb0"]) + (((data["1__kurtosis_y"]) + (data["detected_mjd_diff"]))))))) + (data["flux_d1_pb0"]))))

    def GP_class_16(self,data):
        return (-1.007018 +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (((((data["flux_ratio_sq_sum"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["flux_skew"]) * 2.0)) + (data["flux_skew"])))))))))) +
                0.100000*np.tanh((((((-1.0*((((data["flux_skew"]) * 2.0))))) - (data["flux_skew"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((-1.0*((((((((((data["flux_skew"]) * 2.0)) + (data["4__skewness_x"]))) + ((1.0)))) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh((-1.0*((((1.0) + (((np.maximum(((data["flux_skew"])), ((data["2__skewness_x"])))) + (((data["flux_skew"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"])))))))))) +
                0.100000*np.tanh((((-1.0*((((data["2__skewness_x"]) * 2.0))))) - (((((data["4__skewness_x"]) + (data["4__skewness_x"]))) + (3.0))))) +
                0.100000*np.tanh((-1.0*((np.where(np.where(np.where(2.718282 > -1, data["3__skewness_x"], data["distmod"] ) > -1, 2.718282, data["distmod"] ) > -1, 3.0, data["2__skewness_x"] ))))) +
                0.100000*np.tanh((-1.0*((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) + (((data["detected_flux_mean"]) + (((data["flux_by_flux_ratio_sq_skew"]) + (((2.0) + (data["flux_ratio_sq_sum"])))))))))))) +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (data["detected_flux_by_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((-2.0) - (data["2__skewness_x"]))) - (((data["flux_by_flux_ratio_sq_skew"]) - (((((-2.0) + (data["2__skewness_x"]))) - (data["2__skewness_x"]))))))) +
                0.100000*np.tanh(((np.where(((data["flux_d0_pb2"]) + (((data["3__skewness_x"]) * 2.0))) > -1, -2.0, -2.0 )) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((-2.0) - (((((data["2__skewness_x"]) + (data["2__skewness_x"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))))))) - (2.718282))) +
                0.100000*np.tanh((-1.0*((((((data["flux_skew"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["3__skewness_x"])))))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_skew"]) - (data["flux_skew"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(-2.0 > -1, -2.0, np.where(((-2.0) - (data["2__skewness_x"]))<0, -2.0, ((-2.0) - (data["2__skewness_x"])) ) )) +
                0.100000*np.tanh(((((-3.0) - (((data["flux_skew"]) + (data["flux_skew"]))))) + (((data["flux_skew"]) - (data["flux_skew"]))))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"] > -1, -2.0, (((((((-2.0) > (-2.0))*1.)) - (data["2__skewness_x"]))) - (data["flux_skew"])) )) +
                0.100000*np.tanh(((((-2.0) - (np.where(-2.0 > -1, ((-2.0) - (data["2__skewness_x"])), data["2__skewness_x"] )))) * 2.0)) +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (np.where(data["flux_by_flux_ratio_sq_skew"] > -1, 2.718282, data["5__fft_coefficient__coeff_1__attr__abs__x"] ))))))) +
                0.100000*np.tanh((-1.0*((((((data["detected_flux_ratio_sq_sum"]) + (((data["2__skewness_x"]) / 2.0)))) + (data["4__skewness_x"])))))) +
                0.100000*np.tanh(((((-3.0) - (data["2__skewness_x"]))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_err_min"]) + (((data["flux_median"]) + (data["flux_err_min"]))))))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["detected_flux_err_std"]))) * 2.0)) +
                0.100000*np.tanh(((((-3.0) - (data["4__skewness_x"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_err_min"]) - (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))) )) +
                0.100000*np.tanh(((np.where(np.minimum(((-2.0)), ((data["flux_median"])))<0, ((-2.0) - (data["1__skewness_x"])), (-1.0*((data["4__skewness_x"]))) )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((((data["detected_flux_err_min"]) - (((np.maximum(((data["3__skewness_x"])), ((data["flux_median"])))) + (data["1__skewness_x"])))))), ((-1.0)))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (data["flux_median"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) + (data["flux_err_min"]))) + (data["flux_err_min"]))) + (data["flux_err_min"]))) +
                0.100000*np.tanh((-1.0*((((data["2__skewness_x"]) - (np.minimum(((((-2.0) - (data["detected_flux_by_flux_ratio_sq_skew"])))), ((-2.0))))))))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_median"])), ((data["detected_mjd_diff"])))) - (data["flux_skew"]))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, -3.0, ((((data["4__kurtosis_x"]) - (data["2__skewness_x"]))) - (data["2__skewness_x"])) )) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) + (((data["flux_err_min"]) - (data["flux_err_min"]))))) + (data["flux_d1_pb0"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((((((data["flux_err_min"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((-2.0)), ((((data["2__skewness_x"]) - (data["3__skewness_x"])))))) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_median"]) - (((data["4__skewness_x"]) - (((data["flux_skew"]) - (data["2__skewness_x"]))))))) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((-1.0)), ((data["flux_ratio_sq_skew"])))) - (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.minimum(((((data["detected_flux_err_min"]) + (data["flux_d0_pb0"])))), ((np.where(data["5__kurtosis_y"]<0, data["flux_d0_pb0"], data["detected_flux_err_min"] ))))) +
                0.100000*np.tanh(np.where(((data["flux_err_min"]) / 2.0)>0, data["detected_flux_err_min"], ((data["detected_mjd_diff"]) - (data["flux_err_min"])) )) +
                0.100000*np.tanh(((np.minimum(((data["flux_median"])), ((((data["flux_median"]) - (data["detected_flux_min"])))))) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, ((np.minimum(((data["3__skewness_x"])), ((((-3.0) - (data["3__skewness_x"])))))) * 2.0), data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((data["flux_err_min"]) + (np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["detected_flux_err_min"], (-1.0*((data["flux_min"]))) )))) +
                0.100000*np.tanh((((((((((data["flux_skew"]) - (data["flux_skew"]))) - (data["flux_skew"]))) > (data["1__skewness_x"]))*1.)) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_median"])))))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_err_min"]))))), ((data["flux_err_min"])))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((((np.minimum(((data["flux_err_min"])), ((data["detected_mjd_diff"])))) + (data["flux_err_min"]))) * 2.0)) + (np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_err_min"])))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["flux_median"], np.where(((data["flux_median"]) - (data["detected_flux_min"]))>0, ((data["flux_median"]) - (data["detected_flux_min"])), -3.0 ) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["detected_flux_diff"])), ((data["flux_err_min"]))))))) +
                0.100000*np.tanh(((((-1.0) - (np.where(data["1__skewness_x"]>0, (-1.0*((((-1.0) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))), data["detected_flux_skew"] )))) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((((np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["0__kurtosis_y"]) + (data["detected_mjd_diff"])), data["flux_ratio_sq_skew"] )) + (data["flux_err_min"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"] > -1, data["flux_d0_pb5"], (-1.0*((data["2__skewness_x"]))) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, ((data["flux_median"]) + (((np.minimum(((data["detected_flux_ratio_sq_skew"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) + (0.0)))), data["flux_d0_pb0"] )) +
                0.100000*np.tanh(((np.where(((data["2__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_flux_skew"])) > -1, -2.0, data["flux_ratio_sq_skew"] )) - (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((((-3.0) + (((-3.0) * (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__skewness_x"])))))))), ((2.718282)))) +
                0.100000*np.tanh(((((data["5__kurtosis_y"]) - ((((((data["1__skewness_x"]) - (data["flux_d0_pb5"]))) > (data["2__fft_coefficient__coeff_0__attr__abs__y"]))*1.)))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((data["flux_err_skew"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_ratio_sq_skew"])), ((data["0__kurtosis_x"])))) + (np.minimum(((data["detected_flux_std"])), ((data["flux_ratio_sq_sum"])))))) +
                0.100000*np.tanh(np.where(data["detected_flux_mean"]>0, -3.0, np.where(-3.0>0, data["detected_flux_mean"], data["detected_flux_std"] ) )) +
                0.100000*np.tanh(np.where((((np.where(-2.0 > -1, ((-3.0) - (data["4__fft_coefficient__coeff_0__attr__abs__x"])), data["1__fft_coefficient__coeff_0__attr__abs__y"] )) > (data["flux_err_min"]))*1.) > -1, data["flux_err_min"], data["mjd_diff"] )) +
                0.100000*np.tanh(((-2.0) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_d0_pb0"]) + (np.where(data["flux_err_min"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb3"] )))))/2.0)))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) + (((data["detected_mjd_diff"]) - (data["3__skewness_x"]))))) + (((data["flux_median"]) - (data["flux_median"]))))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(((data["flux_median"]) + ((((((data["flux_median"]) + (data["detected_mjd_diff"]))/2.0)) - (data["2__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))) - (((data["flux_dif2"]) - (data["2__skewness_x"]))))) * 2.0)) +
                0.100000*np.tanh(((((-3.0) + (((np.tanh(((-1.0*((data["detected_flux_by_flux_ratio_sq_sum"])))))) * 2.0)))) / 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_min"] > -1, ((data["flux_d0_pb5"]) - (data["flux_mean"])), data["flux_mean"] )) - (((((data["detected_flux_min"]) * 2.0)) - (data["flux_mean"]))))) +
                0.100000*np.tanh(((np.where(((data["flux_err_min"]) / 2.0)>0, data["flux_err_min"], ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0) )) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((np.minimum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_mjd_diff"])))) + ((((-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["4__kurtosis_y"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_flux_err_min"])), ((data["flux_err_min"]))))), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((np.where(data["distmod"]>0, data["flux_median"], ((((data["flux_median"]) - (data["detected_flux_err_max"]))) * 2.0) )) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.tanh((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((data["flux_mean"]) > (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0)))*1.)) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]>0, -3.0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, data["1__fft_coefficient__coeff_1__attr__abs__x"], -3.0 ) )) +
                0.100000*np.tanh((((((-1.0*(((((data["flux_d1_pb5"]) < (data["detected_flux_median"]))*1.))))) - (data["detected_flux_w_mean"]))) - (data["detected_flux_median"]))) +
                0.100000*np.tanh(((np.where((-1.0*((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_ratio_sq_sum"])))))>0, data["flux_ratio_sq_skew"], data["0__kurtosis_x"] )) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.tanh((data["flux_ratio_sq_skew"]))) + ((((data["detected_mjd_diff"]) + (data["2__kurtosis_y"]))/2.0)))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["detected_mjd_diff"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((((((((data["flux_max"]) + (data["flux_ratio_sq_skew"]))/2.0)) * 2.0)) + (data["detected_mjd_diff"]))/2.0)) + (data["flux_ratio_sq_skew"]))/2.0)) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["detected_flux_std"]) + (data["detected_flux_std"]))))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.where(data["flux_ratio_sq_skew"]<0, data["detected_mjd_diff"], data["flux_err_min"] ))), ((data["flux_max"]))))))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), (((((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_max"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))/2.0))))) +
                0.100000*np.tanh(np.where(((data["flux_dif3"]) - (data["1__skewness_x"]))>0, (-1.0*((data["detected_flux_median"]))), data["flux_d0_pb0"] )) +
                0.100000*np.tanh(((np.where(data["flux_ratio_sq_skew"] > -1, ((data["detected_mjd_diff"]) - (data["detected_flux_err_std"])), data["flux_err_min"] )) + (data["flux_err_min"]))) +
                0.100000*np.tanh((((data["flux_ratio_sq_skew"]) + ((((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))/2.0)))/2.0)) +
                0.100000*np.tanh(((data["mjd_size"]) + (np.where(((data["detected_flux_err_max"]) + (((data["flux_ratio_sq_skew"]) + (data["detected_flux_err_min"])))) > -1, data["detected_flux_err_min"], data["flux_skew"] )))) +
                0.100000*np.tanh((((data["4__kurtosis_x"]) + ((((((data["5__skewness_y"]) + (data["flux_err_min"]))) + (data["detected_flux_max"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.minimum((((((((data["flux_by_flux_ratio_sq_skew"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["2__fft_coefficient__coeff_0__attr__abs__x"])))), ((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["detected_flux_err_min"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]<0, data["flux_err_min"], ((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__skewness_y"])))) - (data["1__skewness_x"])) )) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__x"])), ((np.where(data["flux_diff"]>0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["detected_mjd_diff"] ))))) +
                0.100000*np.tanh((((((-1.0*((((data["flux_d1_pb5"]) - (data["detected_flux_err_mean"])))))) - ((-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((-1.0*((((((np.maximum(((data["flux_d0_pb1"])), ((data["flux_d0_pb0"])))) - (data["ddf"]))) - (data["1__kurtosis_x"])))))) / 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_min"] > -1, data["0__kurtosis_y"], ((data["2__kurtosis_y"]) + (0.0)) )) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_err_min"])), ((((np.minimum(((((data["detected_flux_diff"]) - ((((((data["flux_err_min"]) + (data["flux_err_min"]))/2.0)) / 2.0))))), ((data["detected_flux_err_min"])))) * 2.0))))) +
                0.100000*np.tanh(np.where(data["detected_flux_dif2"] > -1, np.where(data["detected_flux_dif2"]>0, data["2__kurtosis_y"], data["flux_d1_pb0"] ), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((((np.where(data["detected_mjd_diff"] > -1, -3.0, ((data["2__skewness_y"]) * 2.0) )) + (data["3__skewness_y"]))) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (np.where(data["detected_flux_diff"] > -1, np.where(np.minimum(((data["3__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_mjd_diff"]))) > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_ratio_sq_skew"] ), data["flux_median"] )))) +
                0.100000*np.tanh((((((data["mjd_diff"]) - (np.minimum(((data["detected_flux_err_skew"])), ((data["detected_flux_err_mean"])))))) + (data["flux_ratio_sq_skew"]))/2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh((-1.0*((np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, np.maximum(((((data["2__skewness_x"]) - (data["1__kurtosis_x"])))), ((data["2__skewness_x"]))), data["ddf"] ))))) +
                0.100000*np.tanh((((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_err_min"]) + (data["detected_mjd_diff"]))))/2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["flux_err_min"])), ((data["flux_ratio_sq_skew"])))) - (-2.0))) + (((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["1__fft_coefficient__coeff_0__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["flux_dif3"] > -1, ((data["flux_dif3"]) - (data["flux_err_std"])), ((data["4__kurtosis_y"]) * 2.0) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["flux_dif3"])), ((data["0__kurtosis_x"]))))), ((data["flux_dif3"]))))), ((data["flux_dif3"])))) +
                0.100000*np.tanh(((np.where(data["flux_err_min"]>0, data["5__kurtosis_x"], np.minimum(((data["detected_flux_err_min"])), ((data["flux_dif3"]))) )) + (((data["flux_dif3"]) + (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(((((data["detected_flux_err_min"]) + (((data["flux_d0_pb5"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) + ((((data["detected_flux_max"]) + (data["detected_flux_err_min"]))/2.0)))) +
                0.100000*np.tanh((((((((data["detected_flux_err_mean"]) + (data["flux_d1_pb5"]))/2.0)) - (data["2__skewness_x"]))) + ((-1.0*((data["3__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where((-1.0*((data["2__fft_coefficient__coeff_0__attr__abs__y"]))) > -1, data["4__skewness_y"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(np.where((-1.0*((data["detected_flux_std"])))>0, data["4__skewness_y"], data["flux_std"] )>0, data["detected_flux_std"], data["4__skewness_y"] )) / 2.0)) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) - (np.minimum(((data["hostgal_photoz_err"])), ((data["4__fft_coefficient__coeff_1__attr__abs__x"])))))) + (np.tanh((((data["0__kurtosis_x"]) / 2.0)))))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_std"]>0, data["flux_err_skew"], ((data["flux_skew"]) * 2.0) )) +
                0.100000*np.tanh(((data["4__skewness_y"]) + ((((data["4__skewness_y"]) + (data["detected_flux_std"]))/2.0)))) +
                0.100000*np.tanh((((data["0__kurtosis_x"]) + (data["detected_mjd_diff"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), (((((np.maximum(((data["flux_dif3"])), ((data["ddf"])))) + (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d0_pb0"]))))/2.0)))))), ((data["2__fft_coefficient__coeff_0__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"] > -1, ((((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d1_pb2"])))) - (data["flux_err_std"]))) * (data["flux_max"])), data["3__skewness_x"] )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) * (data["detected_flux_err_min"]))))

    def GP_class_42(self,data):
        return (-0.859449 +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((-2.0))))), ((-3.0))))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((-2.0)), ((-2.0)))) +
                0.100000*np.tanh(np.minimum(((-3.0)), ((np.minimum(((np.minimum(((data["flux_mean"])), ((np.minimum(((data["detected_flux_min"])), ((-2.0)))))))), ((-3.0))))))) +
                0.100000*np.tanh(((np.minimum(((((np.minimum(((data["distmod"])), ((data["detected_flux_min"])))) + (data["flux_d1_pb4"])))), ((data["flux_d1_pb3"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["distmod"]) * 2.0))), ((data["detected_flux_min"]))))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_x"])), ((((np.minimum(((data["flux_min"])), ((data["flux_min"])))) * 2.0))))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) / 2.0))), ((np.tanh((-3.0)))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((data["2__skewness_x"]))))), ((data["flux_d0_pb5"])))) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((data["flux_min"])), ((np.minimum(((data["detected_flux_by_flux_ratio_sq_sum"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))))))) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_min"]>0, data["detected_flux_min"], data["flux_min"] ))), ((data["detected_flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_d1_pb2"])), ((np.minimum(((-2.0)), ((data["flux_dif2"]))))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (np.tanh((data["3__skewness_x"]))))) + (np.minimum(((data["detected_flux_min"])), ((((((data["distmod"]) + (data["flux_d0_pb4"]))) * 2.0))))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["distmod"])), ((np.minimum(((data["flux_dif2"])), ((np.where(np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["flux_by_flux_ratio_sq_skew"])))<0, data["flux_diff"], data["distmod"] )))))))) +
                0.100000*np.tanh(((((((((data["detected_flux_min"]) - (data["flux_ratio_sq_skew"]))) - (data["hostgal_photoz_err"]))) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["flux_d0_pb4"])))) + (((data["detected_flux_min"]) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((((data["flux_min"]) + (np.minimum(((data["detected_flux_min"])), ((data["detected_flux_min"])))))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) - (np.minimum(((data["distmod"])), ((data["distmod"]))))))), ((np.minimum(((data["mjd_size"])), ((np.minimum(((data["distmod"])), ((data["distmod"])))))))))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["distmod"])))) + (((data["distmod"]) + (data["1__kurtosis_x"]))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (data["flux_d0_pb5"]))) + (np.minimum(((data["1__kurtosis_x"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((np.where(np.maximum(((data["detected_flux_std"])), (((((data["detected_flux_min"]) > (data["flux_d0_pb4"]))*1.)))) > -1, data["detected_flux_min"], data["detected_flux_ratio_sq_skew"] )) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["5__kurtosis_y"]) + (data["5__kurtosis_y"])))), ((data["detected_flux_min"]))))), ((data["flux_min"])))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["distmod"]))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d1_pb2"]) / 2.0))), ((((data["detected_flux_min"]) + (((data["hostgal_photoz_err"]) + (data["detected_flux_min"])))))))) + (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((((((data["detected_flux_std"]) + (data["flux_diff"]))) / 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) / 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (((((data["detected_flux_min"]) + (((data["flux_min"]) + (data["hostgal_photoz_err"]))))) * 2.0)))) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) - (data["flux_diff"]))) - (data["flux_diff"]))) +
                0.100000*np.tanh(((data["0__kurtosis_x"]) - (np.maximum(((data["5__skewness_x"])), ((data["flux_diff"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["flux_mean"]) + (((data["detected_flux_dif3"]) - (data["flux_max"]))))) - (data["flux_max"]))) +
                0.100000*np.tanh((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)) +
                0.100000*np.tanh((((-1.0*((((data["detected_flux_std"]) - (((data["0__kurtosis_x"]) - (np.where(data["flux_mean"]>0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["detected_flux_std"] ))))))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((data["0__kurtosis_y"]) + (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0), ((((data["flux_d0_pb4"]) + (data["detected_mean"]))) + (data["flux_d0_pb4"])) )) +
                0.100000*np.tanh((((((data["4__skewness_x"]) + (((((data["detected_flux_max"]) + (((data["detected_flux_min"]) * 2.0)))) + (data["detected_flux_min"]))))/2.0)) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((data["1__kurtosis_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((((1.0) > (data["detected_flux_std"]))*1.)) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) + (np.maximum(((data["5__skewness_x"])), ((data["flux_d0_pb5"])))))) +
                0.100000*np.tanh(((((((((((data["detected_flux_ratio_sq_skew"]) - (data["detected_mean"]))) - (data["3__kurtosis_x"]))) - (data["detected_flux_err_std"]))) - (data["3__kurtosis_x"]))) - (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(((data["flux_d0_pb4"]) + (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) < (data["1__skewness_y"]))*1.)) - (data["detected_flux_skew"]))) +
                0.100000*np.tanh(np.tanh((((data["4__kurtosis_y"]) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))))) +
                0.100000*np.tanh(np.tanh((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (np.maximum(((((data["flux_mean"]) - (data["flux_dif3"])))), ((data["5__skewness_x"])))))))))) +
                0.100000*np.tanh((-1.0*((((data["3__kurtosis_x"]) + (data["distmod"])))))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (data["1__kurtosis_y"]))) +
                0.100000*np.tanh(((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((data["0__kurtosis_x"]) + ((((((data["detected_flux_min"]) - (((data["detected_flux_min"]) + (data["3__kurtosis_y"]))))) + (data["distmod"]))/2.0)))/2.0)) +
                0.100000*np.tanh(((((((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__fft_coefficient__coeff_0__attr__abs__x"])))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["detected_flux_skew"]))) +
                0.100000*np.tanh(((np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_dif2"], (((data["detected_mjd_diff"]) > (data["detected_flux_dif2"]))*1.) )) * 2.0)) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_ratio_sq_skew"]))) + (((np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["5__kurtosis_x"]) + (data["flux_by_flux_ratio_sq_skew"])))))) + (data["5__skewness_x"]))))) +
                0.100000*np.tanh(((((((((data["flux_d1_pb0"]) - (data["detected_flux_std"]))) + (data["hostgal_photoz"]))) - (np.tanh((data["flux_d1_pb0"]))))) * 2.0)) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["2__skewness_x"]) * (data["detected_flux_min"]))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]>0, data["4__kurtosis_y"], data["flux_d1_pb0"] )) +
                0.100000*np.tanh((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_diff"]<0, np.maximum(((np.where(data["flux_diff"]<0, data["detected_flux_ratio_sq_skew"], data["5__kurtosis_y"] ))), ((data["detected_flux_ratio_sq_skew"]))), ((data["5__kurtosis_y"]) + (data["hostgal_photoz_err"])) )) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]>0, data["1__kurtosis_x"], ((np.where(data["flux_ratio_sq_skew"] > -1, data["1__kurtosis_x"], (-1.0*((data["1__kurtosis_x"]))) )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh((((np.where(data["flux_d0_pb4"]<0, data["detected_flux_dif2"], data["detected_mjd_diff"] )) > (data["detected_flux_std"]))*1.)) +
                0.100000*np.tanh(((data["detected_mjd_size"]) + (((data["detected_flux_min"]) + (data["flux_diff"]))))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["mjd_diff"]))) +
                0.100000*np.tanh((((data["flux_max"]) < (np.maximum((((((((data["detected_flux_std"]) + (data["0__kurtosis_x"]))) < (data["3__fft_coefficient__coeff_0__attr__abs__y"]))*1.))), ((data["3__kurtosis_x"])))))*1.)) +
                0.100000*np.tanh(((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_d1_pb0"]) + (data["distmod"]))))) +
                0.100000*np.tanh(np.where(((data["detected_mjd_diff"]) - (data["detected_flux_w_mean"]))>0, np.where((((data["hostgal_photoz"]) > (data["hostgal_photoz"]))*1.)<0, data["flux_max"], data["detected_flux_w_mean"] ), data["hostgal_photoz"] )) +
                0.100000*np.tanh(((data["flux_d1_pb1"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["3__kurtosis_y"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_d1_pb5"]))) + (data["2__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(((data["detected_flux_std"]) - (data["detected_flux_std"]))>0, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) - (data["detected_flux_std"])) )) +
                0.100000*np.tanh(np.where(np.where(data["2__skewness_x"]<0, data["2__skewness_x"], np.maximum(((data["2__skewness_x"])), ((data["flux_skew"]))) )<0, -3.0, (13.35520839691162109) )) +
                0.100000*np.tanh(np.where(data["flux_d0_pb0"]<0, np.where(data["flux_d0_pb0"]<0, data["flux_d0_pb0"], data["flux_d0_pb0"] ), ((data["flux_d0_pb1"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((data["flux_mean"]) + (((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_skew"])))) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.maximum(((data["1__kurtosis_x"])), (((((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) > (2.0))*1.)) - (data["4__fft_coefficient__coeff_0__attr__abs__y"])))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + ((((data["flux_d1_pb0"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)))/2.0)))) +
                0.100000*np.tanh(((((((data["0__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) * (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (np.maximum(((((np.maximum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_d0_pb5"])))) + (data["detected_flux_by_flux_ratio_sq_sum"])))), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(np.tanh((data["distmod"]))>0, data["flux_ratio_sq_skew"], np.where(data["hostgal_photoz"]>0, np.where(data["3__kurtosis_y"]>0, data["detected_flux_min"], data["3__fft_coefficient__coeff_0__attr__abs__x"] ), data["detected_flux_diff"] ) )) +
                0.100000*np.tanh(np.where(((data["distmod"]) * 2.0) > -1, np.where(data["hostgal_photoz"]<0, data["hostgal_photoz"], ((data["3__kurtosis_y"]) * 2.0) ), ((np.tanh((data["flux_d0_pb0"]))) * 2.0) )) +
                0.100000*np.tanh((((np.where(data["3__skewness_x"]>0, data["detected_mjd_diff"], data["flux_d0_pb3"] )) > (data["detected_flux_std"]))*1.)) +
                0.100000*np.tanh((((((((((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__kurtosis_y"]))) * 2.0)) < (((data["flux_d1_pb3"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)))))*1.)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["3__skewness_x"]) + (((data["detected_mjd_diff"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]<0, data["4__fft_coefficient__coeff_0__attr__abs__y"], ((((np.where(((data["distmod"]) * 2.0) > -1, data["hostgal_photoz"], data["4__kurtosis_x"] )) * 2.0)) * 2.0) )) * 2.0)) +
                0.100000*np.tanh((((data["distmod"]) + ((((((((((data["detected_flux_min"]) + (data["distmod"]))/2.0)) + (data["distmod"]))/2.0)) + (data["detected_flux_min"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_err_skew"]))) + (data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["2__skewness_y"])) )) +
                0.100000*np.tanh(np.where((((data["flux_by_flux_ratio_sq_sum"]) + (data["2__kurtosis_y"]))/2.0)>0, data["detected_flux_min"], np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]>0, data["3__kurtosis_y"], data["3__kurtosis_y"] ) )) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) * (data["0__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["0__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)))) + ((((data["distmod"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__skewness_x"]))))/2.0)))) +
                0.100000*np.tanh(np.where(data["flux_dif2"]>0, np.where(data["hostgal_photoz"]>0, data["detected_flux_ratio_sq_skew"], data["distmod"] ), data["mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__skewness_x"], ((data["detected_flux_dif3"]) + (data["1__skewness_y"])) )) +
                0.100000*np.tanh(np.where(data["2__skewness_y"]>0, data["flux_d0_pb4"], np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb0"], data["2__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"]>0, data["detected_flux_min"], data["distmod"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_min"]) + (data["1__kurtosis_x"]))) * (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_sum"]) + (data["flux_d1_pb3"]))) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_w_mean"]))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["detected_flux_min"], data["detected_flux_min"] )) +
                0.100000*np.tanh(np.where(data["distmod"]>0, data["5__kurtosis_x"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.maximum((((((data["detected_mjd_diff"]) + (data["detected_flux_err_max"]))/2.0))), ((data["mjd_diff"])))) +
                0.100000*np.tanh(np.minimum(((((((data["flux_err_skew"]) - (np.tanh((data["mwebv"]))))) / 2.0))), ((data["flux_err_skew"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_dif2"] > -1, ((((((data["detected_mjd_diff"]) > (np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif2"] )))*1.)) > (data["detected_flux_dif2"]))*1.), data["detected_flux_dif2"] )) +
                0.100000*np.tanh(((np.where(data["detected_mean"]>0, data["flux_dif2"], data["distmod"] )) + (np.maximum(((data["distmod"])), ((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])))))))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["3__kurtosis_y"], np.where(data["flux_d0_pb0"]>0, np.where(data["flux_d0_pb1"]>0, data["3__kurtosis_y"], data["flux_d0_pb0"] ), data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh((((data["detected_flux_w_mean"]) + (((data["detected_flux_err_std"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"]>0, data["5__skewness_y"], (((data["5__skewness_y"]) > (((((((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) + (data["5__skewness_y"]))/2.0)))*1.) )) +
                0.100000*np.tanh((((((((((data["flux_d1_pb1"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["distmod"]))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"]<0, (((((data["1__skewness_x"]) > (data["1__skewness_x"]))*1.)) / 2.0), data["flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_skew"]<0, data["0__skewness_x"], np.where(data["hostgal_photoz"] > -1, ((data["hostgal_photoz"]) * 2.0), data["hostgal_photoz"] ) )) +
                0.100000*np.tanh((((((((data["flux_d1_pb3"]) - (data["0__kurtosis_y"]))) - (data["0__kurtosis_y"]))) + (((data["flux_d1_pb3"]) * 2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["hostgal_photoz_err"], np.where(data["flux_dif2"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["flux_dif2"] > -1, data["detected_flux_err_median"], data["detected_flux_err_median"] ) ) )) +
                0.100000*np.tanh(np.maximum(((((data["1__kurtosis_x"]) * (data["1__kurtosis_x"])))), ((data["1__kurtosis_x"])))) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_err_std"])), ((data["flux_d1_pb1"])))) +
                0.100000*np.tanh(np.tanh((((((((np.minimum(((((data["detected_mjd_diff"]) / 2.0))), ((data["3__skewness_y"])))) + (data["0__kurtosis_y"]))/2.0)) + (data["0__kurtosis_y"]))/2.0)))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["flux_d0_pb0"], np.where(data["flux_dif2"]>0, data["flux_dif2"], data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]<0, data["5__kurtosis_x"], data["4__fft_coefficient__coeff_0__attr__abs__x"] ), data["mjd_diff"] )) +
                0.100000*np.tanh((((((((data["5__kurtosis_y"]) * ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh((((data["distmod"]) > (data["detected_mjd_diff"]))*1.)) +
                0.100000*np.tanh(np.where(data["2__skewness_y"] > -1, np.where(((data["detected_mjd_diff"]) + (data["2__kurtosis_y"])) > -1, np.maximum(((data["detected_mjd_diff"])), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))), data["detected_flux_dif3"] ), data["1__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], ((((data["4__skewness_x"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d0_pb3"], data["2__kurtosis_x"] ))) )) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) * (((((((data["3__skewness_y"]) + (data["2__kurtosis_y"]))) + (data["3__skewness_y"]))) + (((data["2__kurtosis_y"]) + (data["5__skewness_x"]))))))) +
                0.100000*np.tanh(((np.where(data["flux_dif3"]<0, data["detected_mjd_diff"], ((data["0__kurtosis_y"]) - (data["detected_mjd_diff"])) )) - (data["0__kurtosis_y"]))) +
                0.100000*np.tanh((-1.0*((np.where(((data["hostgal_photoz_err"]) * 2.0) > -1, (-1.0*((np.where(data["detected_flux_w_mean"]>0, data["flux_dif2"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )))), data["flux_diff"] ))))) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_sum"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"] > -1, data["1__skewness_x"], (((data["1__fft_coefficient__coeff_1__attr__abs__x"]) > (data["flux_d1_pb0"]))*1.) )))

    def GP_class_52(self,data):
        return (-1.867467 +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_sum"]) + (((((2.718282) + (data["flux_by_flux_ratio_sq_skew"]))) + (np.tanh((data["flux_by_flux_ratio_sq_sum"]))))))) +
                0.100000*np.tanh(((data["flux_min"]) + (((((data["3__skewness_x"]) + (data["flux_min"]))) + (data["flux_min"]))))) +
                0.100000*np.tanh((((((((((data["flux_min"]) + (((data["flux_min"]) + (data["flux_min"]))))) + (data["flux_d1_pb3"]))) + (data["flux_min"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_min"]) + (((data["flux_min"]) + (data["3__fft_coefficient__coeff_0__attr__abs__x"]))))) + (((data["2__skewness_x"]) + (data["2__skewness_x"]))))) +
                0.100000*np.tanh(np.where(data["flux_min"] > -1, data["flux_min"], ((((data["flux_min"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_by_flux_ratio_sq_skew"])) )) +
                0.100000*np.tanh(((((data["flux_min"]) + (((((data["4__kurtosis_x"]) - (data["flux_diff"]))) * 2.0)))) + (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.where(np.tanh((data["2__fft_coefficient__coeff_1__attr__abs__y"]))>0, data["flux_ratio_sq_skew"], data["2__skewness_x"] )) + (((data["mjd_size"]) - (data["detected_flux_err_mean"]))))) +
                0.100000*np.tanh(np.minimum(((data["3__kurtosis_x"])), ((np.minimum(((data["2__kurtosis_x"])), ((data["distmod"]))))))) +
                0.100000*np.tanh(((((np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["5__kurtosis_x"])), ((data["5__kurtosis_x"]))))))) * 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["flux_min"]) + (((data["5__kurtosis_x"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d0_pb3"]) + (((np.where(data["4__kurtosis_x"]<0, data["flux_ratio_sq_skew"], data["flux_diff"] )) / 2.0))))), ((data["hostgal_photoz_err"])))) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((((data["flux_d0_pb2"]) / 2.0))), ((np.minimum(((data["flux_min"])), ((data["flux_err_skew"]))))))) +
                0.100000*np.tanh(((0.367879) * (data["0__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["flux_ratio_sq_skew"]))) - (((data["distmod"]) * (data["flux_err_std"]))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb3"])), ((np.minimum(((data["2__skewness_x"])), (((((data["2__kurtosis_x"]) < (data["1__kurtosis_x"]))*1.)))))))) +
                0.100000*np.tanh(np.minimum(((((data["4__kurtosis_x"]) * 2.0))), ((data["distmod"])))) +
                0.100000*np.tanh((((((((-1.0*((data["flux_diff"])))) - (data["flux_diff"]))) - (((data["flux_diff"]) - (data["5__kurtosis_x"]))))) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((data["4__kurtosis_x"])), ((data["flux_d0_pb3"]))))))) +
                0.100000*np.tanh(np.minimum(((data["0__kurtosis_x"])), ((data["5__kurtosis_x"])))) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) - (data["detected_flux_err_median"]))) + (((data["flux_std"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_std"]))))))) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb3"])), ((((((data["4__skewness_x"]) - (((data["detected_flux_dif3"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (data["hostgal_photoz_err"]))) + (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["1__skewness_x"]) + (np.where(data["mjd_diff"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb2"] )))) + (((data["flux_ratio_sq_skew"]) + (data["3__skewness_y"]))))) +
                0.100000*np.tanh((((((np.where(data["detected_flux_w_mean"]<0, data["flux_median"], data["flux_max"] )) < (data["flux_max"]))*1.)) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["distmod"]))) + (1.0))) +
                0.100000*np.tanh(((((data["detected_flux_diff"]) - (data["flux_err_max"]))) - (((data["detected_flux_std"]) * 2.0)))) +
                0.100000*np.tanh((((data["flux_diff"]) < (((data["flux_d0_pb2"]) - (data["flux_max"]))))*1.)) +
                0.100000*np.tanh(((((((((-1.0*((np.tanh((data["flux_d0_pb4"])))))) > (data["detected_flux_std"]))*1.)) * 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((0.367879) + (((np.minimum(((((np.tanh((((data["flux_d0_pb2"]) + (data["detected_flux_min"]))))) + (data["flux_d0_pb3"])))), ((data["detected_flux_min"])))) * 2.0)))) +
                0.100000*np.tanh(((((((data["flux_median"]) + (data["flux_median"]))) + (data["flux_median"]))) + (((data["5__kurtosis_y"]) * (data["flux_median"]))))) +
                0.100000*np.tanh(((((data["1__kurtosis_y"]) + (((data["flux_median"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) / 2.0)) / 2.0)) - (data["detected_flux_diff"]))) - (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((((((data["flux_median"]) + (((data["2__kurtosis_x"]) + (data["flux_ratio_sq_skew"]))))) + (np.tanh((data["flux_median"]))))) + (2.718282))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["flux_d1_pb2"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["3__skewness_y"]) - (data["flux_d0_pb0"])), data["distmod"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]<0, data["flux_d0_pb4"], ((data["2__kurtosis_x"]) * 2.0) )) +
                0.100000*np.tanh(((((((data["flux_median"]) * 2.0)) + (((((((data["flux_median"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((((data["hostgal_photoz_err"]) - (data["flux_d0_pb0"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_d0_pb0"]))) - (((data["flux_d0_pb0"]) - (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((np.maximum(((data["distmod"])), ((data["flux_median"])))) + (((1.0) + (data["distmod"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["0__kurtosis_x"])), ((((data["flux_median"]) * 2.0)))))), ((data["4__kurtosis_x"])))) +
                0.100000*np.tanh(((np.tanh((((data["flux_mean"]) - (data["5__fft_coefficient__coeff_0__attr__abs__x"]))))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_median"]) - (data["detected_flux_std"]))) + (data["flux_by_flux_ratio_sq_skew"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((((np.where(data["flux_d0_pb2"]>0, data["flux_d0_pb2"], data["flux_d0_pb2"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (data["detected_flux_diff"]))) +
                0.100000*np.tanh(((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["5__kurtosis_x"] )) - (data["detected_flux_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d0_pb2"]>0, data["5__kurtosis_x"], data["5__kurtosis_x"] )) +
                0.100000*np.tanh(((data["flux_median"]) + ((((np.minimum(((data["5__kurtosis_y"])), ((np.tanh((data["flux_err_max"])))))) + (((data["flux_err_max"]) + (data["1__kurtosis_y"]))))/2.0)))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]>0, data["2__kurtosis_x"], np.where(data["flux_d1_pb1"]>0, data["ddf"], np.where(data["flux_d1_pb1"]>0, data["flux_d1_pb1"], (-1.0*((data["flux_d1_pb1"]))) ) ) )) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) - (((data["detected_flux_std"]) - (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__skewness_x"], data["flux_d1_pb4"] )))))) - (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]>0, ((data["flux_median"]) * 2.0), data["mwebv"] )) +
                0.100000*np.tanh(((((((((((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__kurtosis_x"])))) + (data["2__kurtosis_x"]))) * (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(np.where(data["flux_median"]<0, data["detected_flux_err_mean"], data["detected_flux_err_mean"] ) > -1, ((((data["mwebv"]) * 2.0)) * 2.0), data["3__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb2"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]<0, data["flux_d0_pb0"], (((data["flux_d0_pb0"]) < (data["5__kurtosis_y"]))*1.) )) +
                0.100000*np.tanh(((((((np.where(data["flux_d0_pb0"]>0, data["4__kurtosis_x"], ((data["flux_median"]) - (data["2__kurtosis_x"])) )) - (data["flux_d0_pb0"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((((data["flux_d0_pb2"]) > (data["flux_d1_pb1"]))*1.)) * 2.0)) + (((data["flux_d0_pb2"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(data["flux_dif3"]<0, np.minimum(((((data["flux_ratio_sq_skew"]) + (data["flux_d1_pb2"])))), ((((data["flux_ratio_sq_skew"]) * 2.0)))), data["mwebv"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb2"]>0, data["0__kurtosis_x"], data["4__skewness_y"] )) + (np.maximum(((data["4__skewness_y"])), ((data["flux_err_max"])))))) +
                0.100000*np.tanh(((((((data["mjd_diff"]) * (np.where(data["flux_d0_pb1"]<0, data["3__kurtosis_x"], np.where(data["flux_d0_pb1"]<0, data["mjd_diff"], data["5__fft_coefficient__coeff_0__attr__abs__x"] ) )))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["2__kurtosis_x"]) * (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.where((((data["detected_flux_max"]) < (data["flux_d0_pb2"]))*1.)>0, (((data["flux_d1_pb1"]) < (data["flux_d0_pb2"]))*1.), data["flux_err_std"] )) +
                0.100000*np.tanh(np.minimum(((((((data["flux_median"]) * (data["flux_by_flux_ratio_sq_skew"]))) * ((9.0))))), ((data["flux_d0_pb4"])))) +
                0.100000*np.tanh(np.where(data["flux_err_max"]>0, data["mjd_diff"], np.where(data["mjd_diff"]>0, data["0__kurtosis_x"], data["2__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(((((np.where(data["flux_ratio_sq_skew"]<0, data["flux_err_max"], ((data["flux_d0_pb2"]) + (data["distmod"])) )) * 2.0)) + (data["flux_d0_pb4"]))) +
                0.100000*np.tanh(((((np.where(data["0__skewness_x"]<0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["detected_mean"]>0, data["2__kurtosis_x"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) ) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((((data["detected_flux_diff"]) > (data["flux_dif2"]))*1.)) - (((data["detected_flux_diff"]) - (data["flux_ratio_sq_skew"]))))) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * (((data["detected_flux_err_median"]) + (((((data["detected_flux_err_median"]) + (data["2__skewness_y"]))) + (data["2__skewness_y"]))))))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb1"]<0, ((data["flux_d0_pb0"]) * 2.0), data["2__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, data["detected_flux_min"], np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["4__fft_coefficient__coeff_1__attr__abs__y"] ) )) +
                0.100000*np.tanh(((((((((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_d0_pb0"]))) * 2.0)) - (data["2__skewness_y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)>0, data["4__kurtosis_x"], data["flux_d0_pb4"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]>0, np.where(data["0__skewness_x"]>0, ((data["flux_d1_pb0"]) + (data["flux_err_max"])), ((data["flux_err_max"]) + (data["detected_flux_err_median"])) ), data["3__skewness_y"] )) +
                0.100000*np.tanh(((((data["2__kurtosis_x"]) * (((((((data["5__kurtosis_x"]) * (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * (((data["2__skewness_x"]) / 2.0)))))) * 2.0)) +
                0.100000*np.tanh((((((data["flux_median"]) > (data["2__skewness_y"]))*1.)) + (np.where(data["distmod"]>0, data["0__kurtosis_y"], ((data["distmod"]) + (data["flux_median"])) )))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((np.minimum(((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["4__kurtosis_x"]))))), ((data["2__kurtosis_x"]))))), ((data["flux_d1_pb1"])))) * 2.0)) * (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_std"]<0, data["flux_d0_pb3"], np.where(data["flux_median"]<0, data["0__kurtosis_x"], data["1__kurtosis_y"] ) )) +
                0.100000*np.tanh(((((((np.where(data["detected_flux_std"]>0, np.where(data["detected_flux_min"]<0, (-1.0*((data["3__fft_coefficient__coeff_0__attr__abs__y"]))), data["detected_flux_skew"] ), data["3__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["4__skewness_y"]) - (((data["4__kurtosis_x"]) + (data["4__kurtosis_x"])))), data["4__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]>0, data["4__kurtosis_x"], np.where(data["distmod"]>0, np.where(data["hostgal_photoz_err"]<0, data["detected_mean"], data["hostgal_photoz_err"] ), data["0__skewness_x"] ) )) +
                0.100000*np.tanh(np.where(data["flux_std"]>0, np.where(data["hostgal_photoz_err"]>0, data["distmod"], (-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__x"]))) ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(np.where(np.where(data["flux_err_median"]<0, data["detected_mean"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )<0, data["flux_d1_pb0"], data["2__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]>0, np.where(np.tanh((data["4__kurtosis_y"]))>0, data["2__fft_coefficient__coeff_0__attr__abs__y"], data["1__kurtosis_y"] ), ((data["flux_d1_pb4"]) + (data["detected_mean"])) )) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, np.where(data["1__kurtosis_y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], ((data["1__skewness_x"]) + (data["flux_median"])) ), data["1__kurtosis_y"] )) +
                0.100000*np.tanh(((data["2__kurtosis_y"]) * (np.where(np.tanh((data["2__skewness_y"])) > -1, data["flux_diff"], data["2__skewness_y"] )))) +
                0.100000*np.tanh(((data["flux_d1_pb4"]) + (np.where(data["detected_flux_err_mean"] > -1, np.where(data["0__kurtosis_x"] > -1, data["4__kurtosis_y"], data["flux_d1_pb4"] ), data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) - (data["3__skewness_x"]))) * (((data["2__skewness_y"]) + (np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((data["2__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0), data["3__skewness_x"] )))))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["2__kurtosis_x"]) * (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["distmod"]))) * 2.0)))) * 2.0)))))) +
                0.100000*np.tanh(((data["flux_err_max"]) + (((((np.where(data["detected_mean"]>0, data["2__kurtosis_x"], ((data["flux_err_median"]) + (data["2__kurtosis_x"])) )) * 2.0)) + (data["detected_mean"]))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_mean"]<0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__x"], data["flux_ratio_sq_skew"] ), data["flux_ratio_sq_skew"] ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, ((data["detected_flux_skew"]) * (np.where(data["detected_flux_min"]>0, data["detected_flux_ratio_sq_skew"], data["detected_flux_err_median"] ))), ((data["detected_flux_err_median"]) * (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"]<0, data["flux_median"], np.where(data["flux_median"]<0, data["1__skewness_x"], data["5__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["3__skewness_y"], np.where(data["3__skewness_y"]<0, data["0__skewness_x"], np.where(data["0__kurtosis_x"]>0, data["0__kurtosis_x"], data["2__kurtosis_x"] ) ) )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["detected_flux_err_mean"] > -1, np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), (((4.0)))), (4.0) ), (4.0) )) +
                0.100000*np.tanh(((((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["distmod"], data["detected_flux_err_max"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"] > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["detected_flux_err_skew"] ), ((data["detected_flux_err_skew"]) * 2.0) )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, ((data["3__kurtosis_x"]) * (((((data["flux_d0_pb0"]) * 2.0)) * 2.0))), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb0"]>0, data["3__kurtosis_x"], ((((data["4__skewness_y"]) - (data["3__kurtosis_x"]))) - (data["3__kurtosis_x"])) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["mjd_size"]>0, ((np.where(data["mjd_size"]>0, data["1__kurtosis_y"], ((data["detected_flux_err_min"]) + (data["flux_err_mean"])) )) + (data["flux_err_mean"])), data["3__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(data["0__skewness_x"]>0, ((np.where(data["0__skewness_x"]>0, data["hostgal_photoz_err"], data["flux_d1_pb2"] )) - (data["flux_d1_pb2"])), data["flux_d1_pb2"] )) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0), np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb2"], data["5__kurtosis_x"] ) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]<0, ((((data["3__kurtosis_x"]) - (data["4__kurtosis_x"]))) - (data["2__skewness_y"])), data["2__skewness_y"] )) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_mean"] > -1, (((((data["flux_err_max"]) > (data["2__fft_coefficient__coeff_0__attr__abs__x"]))*1.)) - (data["mjd_size"])), data["2__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh((-1.0*((((np.where(data["0__skewness_x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["2__skewness_y"] ), data["4__kurtosis_y"] )) * (((data["2__skewness_y"]) * 2.0))))))) +
                0.100000*np.tanh(((((data["2__kurtosis_x"]) * (((((data["flux_d1_pb2"]) + (data["distmod"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]<0, np.where(data["detected_flux_max"]<0, data["detected_flux_max"], data["detected_flux_max"] ), ((((((data["detected_flux_max"]) < (data["detected_mjd_size"]))*1.)) < (data["flux_d0_pb1"]))*1.) )) +
                0.100000*np.tanh(((np.where(data["flux_err_max"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["distmod"]<0, data["detected_flux_err_skew"], (((-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0) ) )) * 2.0)) +
                0.100000*np.tanh((((11.19984531402587891)) * (((data["distmod"]) + ((((data["flux_w_mean"]) < ((((data["5__kurtosis_x"]) > ((((data["flux_w_mean"]) > (data["5__kurtosis_x"]))*1.)))*1.)))*1.)))))) +
                0.100000*np.tanh(((np.where(data["0__kurtosis_x"]<0, data["flux_err_max"], np.where(data["0__kurtosis_x"]<0, data["detected_flux_err_std"], np.where(data["flux_err_max"]<0, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_err_std"] ) ) )) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_median"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * (data["hostgal_photoz"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((((np.where(data["flux_ratio_sq_skew"]<0, data["3__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, data["distmod"], data["2__fft_coefficient__coeff_0__attr__abs__y"] ) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]>0, np.where(data["detected_mean"]<0, data["2__skewness_y"], data["2__skewness_y"] ), np.where(data["flux_ratio_sq_sum"]>0, (-1.0*((data["2__skewness_y"]))), data["detected_mean"] ) )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["hostgal_photoz_err"], np.maximum(((data["flux_d1_pb0"])), ((np.where(data["flux_ratio_sq_skew"]<0, data["hostgal_photoz_err"], np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_d1_pb0"]))) )))) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d1_pb0"], data["5__fft_coefficient__coeff_0__attr__abs__y"] ) ) )) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["1__skewness_y"], ((((((data["flux_ratio_sq_skew"]) - (data["1__skewness_y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["flux_dif2"])) )) +
                0.100000*np.tanh(((((np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, ((data["flux_ratio_sq_skew"]) * 2.0), ((data["4__skewness_y"]) - (data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_d1_pb2"])))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]>0, data["hostgal_photoz_err"], np.where(data["flux_d1_pb4"] > -1, data["flux_d1_pb4"], data["flux_d1_pb4"] ) )) * 2.0), 3.141593 )) +
                0.100000*np.tanh(((np.where(data["flux_err_skew"]>0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_err_skew"], data["1__fft_coefficient__coeff_0__attr__abs__x"] ), np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__kurtosis_x"] ) )) * 2.0)))

    def GP_class_53(self,data):
        return (-2.781493 +
                0.100000*np.tanh(((data["flux_err_std"]) + (np.minimum(((data["2__skewness_y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_mean"], data["detected_flux_err_median"] )) +
                0.100000*np.tanh((((((((((((data["flux_err_mean"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))/2.0)) + (data["5__skewness_y"]))/2.0)) + (-1.0))/2.0)) + (data["flux_err_std"]))) +
                0.100000*np.tanh(np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["3__skewness_y"])))) +
                0.100000*np.tanh((((data["flux_err_std"]) + (np.maximum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((((data["detected_mean"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))))))))/2.0)) +
                0.100000*np.tanh(((np.minimum(((data["5__skewness_y"])), ((((data["flux_err_mean"]) + (data["detected_mean"])))))) + (np.where(data["detected_flux_diff"] > -1, data["flux_err_mean"], -2.0 )))) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), ((((np.where(data["2__skewness_y"] > -1, data["3__skewness_y"], data["flux_err_max"] )) + (((data["1__kurtosis_x"]) / 2.0))))))) +
                0.100000*np.tanh(((((0.0) + (data["5__skewness_y"]))) + (np.where(-1.0<0, data["flux_err_mean"], data["flux_d1_pb1"] )))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + ((((np.minimum(((data["detected_flux_std"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) + ((((data["5__skewness_y"]) + (data["flux_max"]))/2.0)))/2.0)))) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), (((((data["detected_mean"]) + (data["flux_err_std"]))/2.0))))) +
                0.100000*np.tanh(((((data["flux_err_std"]) + (data["flux_err_std"]))) + (data["3__skewness_y"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (-3.0))) + (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"] > -1, np.where(data["flux_err_std"] > -1, data["flux_err_std"], data["5__fft_coefficient__coeff_1__attr__abs__y"] ), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(data["flux_err_std"]>0, data["flux_err_std"], data["3__skewness_y"] )) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["detected_mean"] > -1, data["flux_err_std"], data["3__skewness_y"] )) +
                0.100000*np.tanh(((((data["5__skewness_y"]) + (((data["5__skewness_y"]) * 2.0)))) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) + (((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((((((data["flux_err_std"]) + (data["flux_err_std"]))) + (((data["3__skewness_y"]) + (((data["1__kurtosis_x"]) / 2.0)))))) + (-2.0))) +
                0.100000*np.tanh(np.minimum(((np.minimum((((((data["3__skewness_y"]) + (data["flux_err_std"]))/2.0))), ((data["detected_mean"]))))), ((((data["4__skewness_y"]) + (data["4__skewness_y"])))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((-3.0)))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((((data["3__skewness_y"]) + (((((-3.0) + (data["3__skewness_y"]))) + (data["flux_err_std"]))))/2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["4__skewness_y"])), ((data["4__skewness_y"]))))), ((((data["3__kurtosis_y"]) + (data["flux_err_mean"])))))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -3.0, ((np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -3.0, data["flux_err_std"] )) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (3.141593)))) )) +
                0.100000*np.tanh(np.minimum(((data["3__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((((np.minimum(((((data["3__kurtosis_y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["3__kurtosis_y"])))) + (data["detected_flux_err_median"]))))))))) +
                0.100000*np.tanh(((((data["3__kurtosis_y"]) + (-2.0))) + (((np.minimum(((data["flux_err_std"])), ((data["flux_err_std"])))) + (-2.0))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((-3.0)), ((-3.0))))))))) +
                0.100000*np.tanh(((data["flux_err_std"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))), ((np.maximum(((data["detected_mean"])), ((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (np.where(data["detected_mean"] > -1, -3.0, ((data["flux_std"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, 3.141593, np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], 3.141593 ) )))) +
                0.100000*np.tanh(np.minimum(((data["flux_err_std"])), ((((np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((np.minimum(((data["3__skewness_y"])), ((data["flux_err_std"])))))))), ((data["3__skewness_y"])))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(((np.where(data["detected_mean"] > -1, data["detected_mean"], np.where(data["detected_flux_err_max"]>0, data["flux_err_std"], data["detected_flux_err_mean"] ) )) - (2.0))) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (((data["flux_max"]) * (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((((((-2.0) + (((-2.0) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["flux_err_std"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_mean"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["flux_err_std"]))))), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], 3.141593 )))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_err_std"]))/2.0)) - (data["flux_err_std"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (2.0))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_err_std"]))) + (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((np.minimum(((data["2__kurtosis_y"])), (((((((data["3__kurtosis_y"]) + (data["2__kurtosis_y"]))/2.0)) + (data["flux_d1_pb1"])))))) + (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (3.0)))), ((data["detected_mjd_diff"])))) +
                0.100000*np.tanh(np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((data["2__kurtosis_y"])), ((((np.minimum(((data["flux_max"])), ((((data["4__skewness_y"]) * 2.0))))) * 2.0)))))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_err_std"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"] > -1, data["flux_d0_pb1"], data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb1"])), ((((data["flux_max"]) - (data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((np.where(data["flux_err_max"]>0, ((data["flux_max"]) - (data["detected_flux_ratio_sq_skew"])), ((data["flux_max"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"])) )) - ((-1.0*((data["flux_err_median"])))))) +
                0.100000*np.tanh(((-2.0) + (np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -2.0, data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["detected_mean"] )))) +
                0.100000*np.tanh(((((data["2__skewness_x"]) - (data["detected_flux_ratio_sq_skew"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((((data["flux_err_mean"]) - (data["3__kurtosis_y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_max"])))) +
                0.100000*np.tanh(np.where(data["flux_max"]<0, data["3__kurtosis_y"], np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["3__kurtosis_y"]))) )) +
                0.100000*np.tanh((((data["4__skewness_y"]) + (np.where(data["3__skewness_x"]>0, np.where(-1.0>0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ), data["flux_d0_pb3"] )))/2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_max"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((((((-1.0*((data["detected_flux_ratio_sq_skew"])))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((np.where(data["0__skewness_x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["0__skewness_x"] ), np.where(data["0__skewness_x"]>0, data["0__skewness_x"], data["0__skewness_x"] ) ))))) +
                0.100000*np.tanh(np.where(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * (data["0__skewness_x"]))<0, ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0), data["0__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((((((np.minimum(((data["flux_d0_pb1"])), ((data["flux_err_mean"])))) + (-2.0))) + (data["detected_mean"])))), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)<0, data["3__kurtosis_y"], ((data["1__kurtosis_x"]) * 2.0) )))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_d0_pb1"]<0, data["flux_d0_pb1"], np.where(data["flux_d0_pb1"]>0, data["detected_mean"], data["flux_d0_pb1"] ) ))), ((np.minimum(((data["detected_mean"])), ((data["2__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["flux_std"]) - (data["5__fft_coefficient__coeff_0__attr__abs__x"])))))), ((((-3.0) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])))))) + (data["flux_std"]))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh((((((data["flux_err_std"]) + (np.tanh((data["flux_d1_pb0"]))))/2.0)) - (((data["flux_d1_pb2"]) * (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["0__kurtosis_x"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) - (data["flux_err_skew"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((((data["flux_d1_pb1"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))))), ((data["detected_mean"])))) +
                0.100000*np.tanh(((((((((((((data["flux_err_median"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.maximum((((((np.where(data["3__kurtosis_y"]>0, data["3__skewness_y"], data["3__kurtosis_y"] )) + ((((data["3__kurtosis_y"]) + (data["detected_flux_diff"]))/2.0)))/2.0))), ((np.tanh((data["detected_mean"])))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_min"]<0, data["3__kurtosis_y"], np.minimum(((3.141593)), ((np.minimum(((-2.0)), (((((data["2__kurtosis_x"]) + (data["1__skewness_y"]))/2.0))))))) )) / 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb1"])), (((((((-1.0*((np.tanh((data["detected_flux_by_flux_ratio_sq_sum"])))))) / 2.0)) / 2.0))))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"] > -1, ((np.where(data["3__kurtosis_y"]>0, data["flux_d0_pb1"], np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, data["0__skewness_y"], data["flux_d0_pb1"] ) )) * 2.0), data["flux_err_std"] )) +
                0.100000*np.tanh((((np.tanh(((((data["ddf"]) + (data["2__skewness_x"]))/2.0)))) + (data["detected_flux_err_max"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__skewness_y"])), ((np.minimum(((data["flux_err_max"])), ((data["3__kurtosis_y"])))))))), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh((-1.0*((np.where(data["0__kurtosis_y"] > -1, ((data["0__skewness_x"]) - (((data["detected_flux_max"]) - (data["0__skewness_x"])))), data["2__fft_coefficient__coeff_1__attr__abs__x"] ))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_max"])), ((data["3__kurtosis_y"]))))), ((((np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_mean"])))) / 2.0))))) +
                0.100000*np.tanh((((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_mjd_diff"]) / 2.0)))/2.0)) * (np.minimum(((data["flux_err_max"])), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(np.minimum(((((-3.0) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((((np.where(-3.0>0, data["2__skewness_x"], data["mjd_diff"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_ratio_sq_sum"]))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.minimum(((((data["1__kurtosis_y"]) + (((data["mjd_diff"]) - (((data["2__skewness_y"]) / 2.0))))))), ((data["detected_flux_max"])))) +
                0.100000*np.tanh(np.minimum(((((data["2__kurtosis_x"]) + (data["3__kurtosis_y"])))), ((data["3__kurtosis_y"])))) +
                0.100000*np.tanh(np.minimum(((((((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) * (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) * (((data["4__skewness_y"]) * 2.0))))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((((((3.141593) < (np.where(data["ddf"]<0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )))*1.)) / 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__skewness_y"]))))), ((((((data["flux_ratio_sq_sum"]) / 2.0)) - (data["flux_dif2"])))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_max"])))), ((((data["4__skewness_y"]) * 2.0))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_flux_w_mean"] > -1, data["0__skewness_x"], data["0__skewness_x"] ))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((np.tanh((np.maximum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["mjd_size"])))))) - (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))))))) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((((data["4__skewness_y"]) / 2.0))), ((((data["flux_d1_pb1"]) - (data["2__skewness_y"])))))) * 2.0))), ((np.minimum(((data["detected_flux_ratio_sq_sum"])), ((data["3__fft_coefficient__coeff_0__attr__abs__x"]))))))) +
                0.100000*np.tanh(((3.0) * ((((data["2__kurtosis_y"]) + (data["detected_mean"]))/2.0)))) +
                0.100000*np.tanh(np.where(np.minimum(((data["4__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["2__kurtosis_y"])), ((np.tanh((data["flux_median"]))))))))>0, ((data["flux_max"]) / 2.0), data["2__skewness_y"] )) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_err_std"])))) +
                0.100000*np.tanh((-1.0*((((((data["2__skewness_x"]) - ((((data["2__kurtosis_y"]) < (data["0__kurtosis_y"]))*1.)))) / 2.0))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))))), ((data["2__skewness_y"]))))), ((data["2__skewness_x"])))) +
                0.100000*np.tanh(((np.where((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) < (data["detected_flux_dif3"]))*1.) > -1, data["2__fft_coefficient__coeff_1__attr__abs__x"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["detected_flux_err_max"], data["2__kurtosis_y"] )) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["detected_flux_std"]))))), ((((data["hostgal_photoz"]) + ((-1.0*((((3.141593) * (data["detected_flux_err_median"]))))))))))) +
                0.100000*np.tanh(np.where(((((3.67651557922363281)) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) > -1, (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_max"]))/2.0), data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.where(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_y"], np.where(data["1__skewness_y"]>0, data["flux_d1_pb2"], data["flux_d1_pb2"] ) )>0, 2.718282, data["flux_d1_pb2"] )) +
                0.100000*np.tanh(np.where(data["flux_mean"] > -1, data["flux_d1_pb1"], data["2__kurtosis_y"] )) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((data["2__skewness_x"]) + (((data["3__kurtosis_y"]) * 2.0))))), ((data["3__kurtosis_y"])))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((2.718282) - (data["detected_flux_by_flux_ratio_sq_sum"]))))))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["detected_mjd_diff"]) + (data["detected_flux_dif2"]))/2.0)))) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb1"] > -1, data["1__kurtosis_x"], data["flux_median"] )) - (((data["0__skewness_y"]) * (data["2__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["flux_max"]) + ((((np.tanh((((data["flux_err_std"]) + (0.0))))) + (data["flux_median"]))/2.0)))) +
                0.100000*np.tanh(((data["5__skewness_y"]) + (data["1__skewness_y"]))) +
                0.100000*np.tanh((((data["detected_flux_mean"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(np.tanh((((data["detected_flux_err_max"]) - ((-1.0*((np.minimum(((((data["detected_flux_err_max"]) / 2.0))), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]))))))))))) +
                0.100000*np.tanh(((np.minimum(((((-3.0) - ((((0.0)) + (2.718282)))))), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["detected_flux_diff"]))) +
                0.100000*np.tanh((((((data["detected_flux_std"]) * (((((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["3__skewness_x"])))) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) + (0.367879))/2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"]<0, data["3__kurtosis_y"], (((((np.minimum(((data["3__kurtosis_y"])), ((data["3__kurtosis_y"])))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) - (((data["flux_err_max"]) * 2.0))) )) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_err_max"])), ((data["detected_mean"])))) +
                0.100000*np.tanh(((np.tanh((data["flux_ratio_sq_sum"]))) - (np.where(data["0__kurtosis_y"]>0, (((data["flux_ratio_sq_skew"]) + (data["4__skewness_y"]))/2.0), data["flux_err_skew"] )))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((((data["flux_mean"]) / 2.0)) - (data["flux_err_std"]))))) +
                0.100000*np.tanh(((data["flux_median"]) - ((((-1.0*((data["5__skewness_x"])))) - (((data["flux_dif2"]) * (data["flux_by_flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(np.where(np.minimum(((data["3__skewness_y"])), ((data["3__fft_coefficient__coeff_1__attr__abs__y"])))<0, np.tanh((((np.tanh((data["3__fft_coefficient__coeff_0__attr__abs__x"]))) / 2.0))), data["1__kurtosis_y"] )) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((((data["2__kurtosis_x"]) / 2.0))), ((data["detected_flux_by_flux_ratio_sq_sum"])))<0, data["flux_d0_pb2"], data["3__kurtosis_y"] ))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((np.where((((data["flux_d1_pb2"]) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)<0, data["5__skewness_y"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ))), ((data["3__skewness_y"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_diff"] > -1, data["2__kurtosis_y"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["3__kurtosis_y"], data["2__skewness_y"] ) )) +
                0.100000*np.tanh(np.where(data["mwebv"] > -1, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["2__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((data["detected_flux_err_median"])))) +
                0.100000*np.tanh((((((((((np.tanh((data["flux_diff"]))) / 2.0)) * 2.0)) + (data["1__kurtosis_x"]))/2.0)) / 2.0)) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"] > -1, (-1.0*((np.minimum(((data["flux_d0_pb1"])), ((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0))))))), -2.0 )))

    def GP_class_62(self,data):
        return (-1.361137 +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["flux_d0_pb5"]) + (((((data["detected_flux_median"]) + (((data["5__skewness_x"]) * 2.0)))) + (data["distmod"])))), -3.0 )) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) + (((((np.maximum(((data["detected_flux_min"])), ((data["detected_flux_min"])))) + (((data["4__kurtosis_x"]) + (data["detected_flux_min"]))))) + (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["2__kurtosis_x"])), ((data["flux_min"]))))), ((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["flux_d1_pb4"]) + (data["5__skewness_x"])))), ((data["5__skewness_x"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((data["flux_d0_pb5"]))))), ((data["flux_d0_pb5"]))))), ((np.minimum(((data["flux_min"])), ((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["5__skewness_x"])))))))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["flux_min"])), ((np.minimum(((data["flux_d0_pb5"])), ((((data["flux_d0_pb5"]) + (data["flux_d0_pb5"])))))))))), ((data["detected_flux_min"]))))), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((1.0)))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_d0_pb1"])), ((np.tanh((data["flux_d0_pb5"]))))))), ((np.minimum(((data["1__skewness_x"])), ((data["1__kurtosis_x"]))))))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((data["0__kurtosis_x"])))) +
                0.100000*np.tanh(((((((data["distmod"]) - (np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))) - (data["detected_flux_err_std"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((((np.minimum(((data["detected_flux_mean"])), ((data["hostgal_photoz_err"])))) - (((data["hostgal_photoz_err"]) - (-1.0))))))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.minimum((((((((data["detected_flux_min"]) + (np.minimum(((data["5__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))/2.0)) + (data["flux_d0_pb4"])))), ((data["4__skewness_x"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["0__skewness_y"])), ((np.minimum(((data["detected_flux_min"])), ((data["detected_mjd_diff"])))))))), ((data["flux_dif2"])))) +
                0.100000*np.tanh(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), ((((data["flux_dif2"]) * 2.0))))) + (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.minimum(((((data["3__kurtosis_x"]) * 2.0))), (((((((data["5__kurtosis_x"]) < (data["4__skewness_x"]))*1.)) / 2.0))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_min"])), ((((data["detected_flux_min"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))))) - (data["flux_err_min"]))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - ((((data["4__skewness_y"]) > (data["flux_dif2"]))*1.)))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(np.minimum(((data["distmod"])), ((data["distmod"]))) > -1, data["flux_d0_pb5"], (((data["distmod"]) + ((((data["distmod"]) + (data["distmod"]))/2.0)))/2.0) )) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], ((data["distmod"]) * 2.0) )) +
                0.100000*np.tanh(((((np.minimum(((((data["hostgal_photoz_err"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["detected_flux_ratio_sq_skew"])))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh((((((data["flux_d0_pb5"]) + (data["flux_d1_pb5"]))/2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["5__skewness_x"]) + (data["hostgal_photoz_err"])))), ((data["2__kurtosis_x"]))))), ((((((data["flux_min"]) + (data["2__kurtosis_x"]))) + (0.367879)))))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(((((((((data["detected_flux_min"]) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["detected_flux_min"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((np.minimum(((((data["2__kurtosis_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))), ((((data["4__kurtosis_x"]) + ((((1.0) + (data["distmod"]))/2.0))))))) + (data["flux_d0_pb5"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_median"])), ((((np.minimum(((((((data["1__kurtosis_x"]) - (data["1__skewness_y"]))) + (data["flux_d0_pb5"])))), ((data["flux_d1_pb5"])))) - (data["1__skewness_y"])))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (np.maximum(((((data["1__kurtosis_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["hostgal_photoz"])))))) - (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["detected_mjd_diff"]) - (data["5__skewness_x"]))))) - (data["detected_mjd_diff"]))) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["3__skewness_x"]) - (((((data["detected_flux_err_std"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__kurtosis_y"]))))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(3.141593 > -1, data["hostgal_photoz_err"], np.minimum(((data["detected_flux_min"])), (((0.28365617990493774)))) )) +
                0.100000*np.tanh(((data["1__kurtosis_x"]) + (((data["flux_d1_pb2"]) * 2.0)))) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) - (np.where(data["flux_d0_pb2"]<0, data["1__skewness_y"], data["2__kurtosis_x"] )))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((data["mjd_size"]) + (((data["distmod"]) - (((data["detected_flux_by_flux_ratio_sq_sum"]) + (data["1__skewness_y"]))))))) - (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, data["2__skewness_x"], ((((data["2__skewness_x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh((-1.0*(((-1.0*(((((((-1.0*((data["mjd_diff"])))) + (data["flux_d0_pb4"]))) - (data["2__skewness_y"]))))))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["flux_d1_pb0"]))) - (data["detected_mjd_diff"]))) - (((data["flux_d0_pb5"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_mjd_diff"]))) * 2.0)) - (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_flux_dif3"]) - ((((data["flux_d1_pb5"]) < (data["2__fft_coefficient__coeff_0__attr__abs__x"]))*1.)))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (data["3__skewness_x"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.where(data["5__skewness_x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_d0_pb5"]) + (data["5__skewness_x"])) )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh((((((data["1__skewness_y"]) < (((data["2__kurtosis_x"]) + (data["detected_mjd_diff"]))))*1.)) - ((((data["detected_mjd_diff"]) > ((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)))*1.)))) +
                0.100000*np.tanh(((((((data["3__kurtosis_x"]) + (data["3__kurtosis_x"]))) + (data["2__kurtosis_x"]))) + (2.718282))) +
                0.100000*np.tanh(np.where(-2.0 > -1, ((data["flux_dif3"]) - (((data["1__skewness_x"]) - (data["detected_flux_by_flux_ratio_sq_skew"])))), ((((data["1__skewness_x"]) - (data["5__kurtosis_y"]))) * 2.0) )) +
                0.100000*np.tanh(((((((((data["detected_flux_dif3"]) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) - ((-1.0*((data["distmod"])))))) +
                0.100000*np.tanh(((((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["5__fft_coefficient__coeff_1__attr__abs__y"] )) - (((2.0) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["1__skewness_y"])))))) < ((((data["1__skewness_y"]) < (data["0__fft_coefficient__coeff_0__attr__abs__y"]))*1.)))*1.)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["5__skewness_x"]) * 2.0)) - (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((np.maximum(((data["mjd_diff"])), ((data["2__skewness_x"])))) - (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) * 2.0)) < (data["flux_d0_pb5"]))*1.)) - (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["flux_ratio_sq_sum"]))) - (((((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (((data["2__skewness_x"]) + (((((((np.tanh((data["flux_ratio_sq_sum"]))) / 2.0)) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh((((((data["flux_w_mean"]) < (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_d0_pb5"]) * 2.0) )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (data["1__skewness_y"]))) + (data["1__kurtosis_x"]))) - (data["1__skewness_y"]))) +
                0.100000*np.tanh(((((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) * (data["detected_flux_dif3"]))) + (np.minimum(((data["3__kurtosis_x"])), ((data["3__kurtosis_x"])))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["flux_dif2"]))/2.0)) + (data["flux_dif2"]))/2.0)) +
                0.100000*np.tanh((((((data["flux_d0_pb2"]) < (data["flux_mean"]))*1.)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d0_pb0"])))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_dif3"]) + (((data["detected_flux_dif3"]) + (((data["detected_flux_dif3"]) - (((data["flux_max"]) * 2.0)))))))) +
                0.100000*np.tanh(((((data["0__kurtosis_y"]) + (((data["3__kurtosis_y"]) + (data["detected_flux_err_min"]))))) + ((-1.0*((data["4__skewness_y"])))))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (data["2__skewness_x"]))) + (((((data["2__skewness_x"]) * 2.0)) + (np.where(data["2__skewness_x"]<0, data["2__skewness_x"], data["2__skewness_x"] )))))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_min"] > -1, np.tanh((data["detected_flux_dif3"])), data["detected_flux_dif3"] )) +
                0.100000*np.tanh(np.where(np.where(np.where(data["4__kurtosis_x"]<0, data["detected_flux_dif3"], data["flux_d0_pb0"] )<0, data["flux_d0_pb0"], data["mwebv"] ) > -1, data["flux_d0_pb0"], ((data["flux_d0_pb0"]) * 2.0) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, ((data["flux_d0_pb5"]) + (data["detected_flux_err_max"])), np.where(data["5__skewness_x"]<0, data["detected_flux_dif3"], ((data["detected_flux_err_min"]) + (data["0__kurtosis_y"])) ) )) +
                0.100000*np.tanh(((np.where(((data["mjd_size"]) - (((data["mwebv"]) - (data["5__kurtosis_y"]))))<0, data["3__kurtosis_y"], data["3__kurtosis_y"] )) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["1__skewness_x"])), ((((data["detected_flux_dif3"]) * (data["detected_flux_dif3"])))))) * (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((np.where(((data["flux_d0_pb1"]) - (data["flux_median"])) > -1, data["flux_d1_pb1"], data["flux_d0_pb1"] )) > (data["flux_d0_pb1"]))*1.)) - (data["flux_median"]))) +
                0.100000*np.tanh(((((((((((((data["detected_flux_dif3"]) + (data["flux_dif3"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["flux_dif3"]))) +
                0.100000*np.tanh(((((((np.where(((data["detected_flux_max"]) * 2.0)>0, ((data["flux_d1_pb4"]) - (data["flux_d0_pb1"])), data["detected_flux_err_min"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_d1_pb2"])))) +
                0.100000*np.tanh((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )))*1.)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, np.where(data["1__skewness_x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["3__kurtosis_x"] ), data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((((((data["detected_flux_dif3"]) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["flux_median"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_err_min"]) + (np.maximum(((((((data["detected_flux_err_min"]) + (data["hostgal_photoz"]))) + (np.maximum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_flux_err_min"]))))))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((data["flux_diff"]) < (data["1__skewness_x"]))*1.), data["detected_flux_by_flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb3"]>0, data["4__kurtosis_x"], np.where(data["flux_d1_pb4"]>0, np.where(data["flux_d1_pb3"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["flux_err_skew"] ), data["4__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(np.where(data["distmod"]<0, data["distmod"], data["mwebv"] )<0, ((data["detected_flux_dif3"]) + (data["0__kurtosis_y"])), data["distmod"] )) +
                0.100000*np.tanh((((data["distmod"]) + (np.where(data["hostgal_photoz"]>0, (((data["distmod"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.), data["mwebv"] )))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]<0, data["detected_flux_err_median"], data["flux_max"] )) +
                0.100000*np.tanh(((((data["0__skewness_x"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) * (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"] > -1, data["detected_flux_dif3"], data["detected_mjd_size"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_dif3"] > -1, ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["detected_flux_ratio_sq_skew"])), ((np.where(data["flux_mean"]<0, data["flux_ratio_sq_skew"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) * (data["3__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"]>0, (((data["mjd_size"]) < (data["hostgal_photoz"]))*1.), (((data["mwebv"]) + (np.minimum(((data["hostgal_photoz_err"])), ((data["1__skewness_x"])))))/2.0) )) +
                0.100000*np.tanh(((((((data["flux_d1_pb3"]) * 2.0)) * (np.where(np.where(data["flux_d1_pb3"] > -1, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb4"] ) > -1, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb3"] )))) * 2.0)) +
                0.100000*np.tanh(np.maximum(((data["flux_d0_pb0"])), ((((data["flux_d0_pb0"]) * 2.0))))) +
                0.100000*np.tanh(((((((((data["0__kurtosis_x"]) - (data["detected_mean"]))) - (((data["mjd_size"]) * 2.0)))) - (((data["0__kurtosis_x"]) * 2.0)))) - (data["5__kurtosis_y"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["flux_d0_pb0"], (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) > (np.where((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (data["3__kurtosis_x"]))*1.) > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["3__kurtosis_x"] )))*1.) )) +
                0.100000*np.tanh(((((((((((np.where(data["detected_mean"]>0, data["detected_flux_median"], data["2__skewness_y"] )) - (data["flux_median"]))) * 2.0)) * 2.0)) - (data["flux_median"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["flux_d0_pb5"]<0, ((data["distmod"]) + (data["3__fft_coefficient__coeff_0__attr__abs__x"])), data["detected_flux_dif3"] ) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_median"]))) + (((data["detected_flux_err_median"]) + (((data["2__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["distmod"] > -1, (-1.0*((data["flux_median"]))), data["distmod"] ), ((data["hostgal_photoz_err"]) + (data["flux_median"])) )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb0"]<0, data["2__skewness_x"], np.where(data["flux_d0_pb0"]<0, data["flux_ratio_sq_skew"], (7.80123519897460938) ) )) * 2.0)) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) * (data["4__kurtosis_x"]))) * (((data["flux_d0_pb5"]) * (data["1__skewness_x"]))))) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))<0, ((data["mwebv"]) - (data["3__kurtosis_x"])), np.where(data["distmod"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh(((((data["detected_flux_dif3"]) - (data["flux_median"]))) + (((data["detected_flux_dif3"]) - (((data["detected_flux_dif3"]) * (((data["detected_flux_dif3"]) - (data["flux_median"]))))))))) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], ((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, np.where(data["hostgal_photoz"]<0, np.where(data["detected_flux_err_skew"]<0, data["hostgal_photoz_err"], data["hostgal_photoz"] ), data["hostgal_photoz_err"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(np.where(np.where(data["flux_skew"]>0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["flux_skew"] )>0, data["distmod"], ((data["flux_skew"]) + (data["0__skewness_x"])) )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["detected_flux_err_min"], data["3__kurtosis_y"] )) +
                0.100000*np.tanh(((((np.where(data["hostgal_photoz_err"]>0, data["distmod"], ((((data["hostgal_photoz_err"]) + (data["flux_d1_pb4"]))) + (data["flux_d1_pb5"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_median"]>0, data["flux_err_min"], ((np.where(data["detected_flux_err_median"]>0, data["flux_err_min"], data["2__skewness_x"] )) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.) )) +
                0.100000*np.tanh(((data["detected_flux_err_max"]) - (data["0__skewness_y"]))) +
                0.100000*np.tanh((((((data["detected_flux_err_std"]) < ((((data["detected_mjd_size"]) > (((data["flux_by_flux_ratio_sq_sum"]) - (np.maximum(((data["flux_mean"])), ((data["detected_flux_err_skew"])))))))*1.)))*1.)) - (data["detected_flux_err_std"]))) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) < (((((((data["detected_mjd_size"]) < ((((data["2__kurtosis_y"]) < ((((data["detected_mjd_diff"]) < (data["detected_flux_min"]))*1.)))*1.)))*1.)) < (data["4__fft_coefficient__coeff_0__attr__abs__x"]))*1.)))*1.)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) - (data["1__skewness_x"])))), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["flux_dif2"]>0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__kurtosis_y"], data["0__fft_coefficient__coeff_1__attr__abs__x"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (data["1__fft_coefficient__coeff_1__attr__abs__x"]))*1.), data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["3__skewness_x"])))) +
                0.100000*np.tanh(np.where(np.where(data["0__skewness_x"] > -1, (((data["flux_d0_pb2"]) < (data["flux_d0_pb2"]))*1.), data["flux_d0_pb2"] )<0, data["detected_mjd_size"], (((data["flux_d0_pb2"]) < (data["flux_mean"]))*1.) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_diff"])), ((data["2__skewness_x"]))))), ((np.minimum(((np.minimum(((data["flux_d1_pb1"])), ((data["distmod"]))))), ((data["flux_d1_pb1"]))))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, (((data["5__skewness_x"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.), data["detected_mjd_size"] )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]>0, np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_err_skew"], np.where(data["detected_flux_err_median"]>0, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ) ), data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.where(data["hostgal_photoz_err"]<0, np.where(data["flux_d0_pb4"]<0, data["hostgal_photoz_err"], data["detected_mjd_size"] ), data["detected_mjd_size"] ), data["hostgal_photoz_err"] )) * 2.0)) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) < ((((((data["detected_mjd_diff"]) < (data["detected_mjd_diff"]))*1.)) * 2.0)))*1.)) +
                0.100000*np.tanh(np.where(((data["flux_dif2"]) * (data["flux_dif2"]))>0, data["1__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] ) )))

    def GP_class_64(self,data):
        return (-2.164979 +
                0.100000*np.tanh(((((((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) + (((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) * 2.0)))) - (((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((-1.0*((((data["detected_mjd_diff"]) * 2.0))))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["flux_err_max"]) - (np.where(((((data["detected_mjd_diff"]) * 2.0)) * 2.0) > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) - (np.tanh((data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))/2.0)) + ((-1.0*((data["detected_mjd_size"])))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) * 2.0)))) +
                0.100000*np.tanh((((((-1.0*((np.where((-1.0*((data["detected_mjd_diff"])))>0, data["detected_mjd_diff"], data["detected_mjd_diff"] ))))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((((((np.tanh((data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) / 2.0)))) +
                0.100000*np.tanh(((((np.maximum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))) * 2.0)) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) > (data["1__skewness_y"]))*1.)) - ((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))/2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.where(((data["flux_ratio_sq_skew"]) * 2.0) > -1, (-1.0*((data["detected_mjd_diff"]))), ((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"])) )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) * 2.0)))) + (data["flux_ratio_sq_skew"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.minimum((((-1.0*((data["detected_mjd_diff"]))))), (((((((-1.0*((data["detected_mjd_diff"])))) * 2.0)) + (data["flux_ratio_sq_skew"])))))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_flux_mean"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_ratio_sq_skew"]) - (data["detected_mjd_size"]))) - (((data["detected_mjd_size"]) - (data["flux_d0_pb1"]))))) - (data["detected_mjd_size"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((((1.0) + (((data["detected_mjd_diff"]) + (data["detected_mjd_diff"])))))))) +
                0.100000*np.tanh((-1.0*((((((((1.0) + (data["detected_mjd_diff"]))) * 2.0)) + (((((data["detected_mjd_diff"]) * 2.0)) + (1.0)))))))) +
                0.100000*np.tanh((-1.0*((((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) - (((data["detected_mjd_diff"]) * 2.0))))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) * 2.0)))) + (((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_sum"]))) * 2.0)))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_ratio_sq_skew"])), ((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"])))))) + (data["flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_mjd_size"]))) - (data["detected_mjd_size"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((-1.0*((np.where((((((((data["detected_mjd_diff"]) / 2.0)) + (data["flux_median"]))/2.0)) * 2.0)<0, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) * 2.0) ))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_mjd_diff"] > -1, np.where(((data["detected_mjd_diff"]) / 2.0) > -1, 2.718282, data["detected_mjd_diff"] ), data["detected_mjd_diff"] ))))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, -3.0, np.where(data["detected_mjd_diff"] > -1, -3.0, (8.0) ) )) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) - (data["flux_ratio_sq_skew"]))) * 2.0)) + (((np.maximum(((data["detected_flux_std"])), ((data["detected_flux_std"])))) + (data["flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((data["detected_flux_std"]) - (data["detected_flux_diff"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))) + (((data["detected_flux_dif2"]) * 2.0)))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh(((((((-1.0) * 2.0)) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_std"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) - (data["flux_d0_pb4"]))) - (data["ddf"]))) +
                0.100000*np.tanh(((((((data["detected_flux_std"]) - (data["flux_d0_pb4"]))) - (data["flux_d0_pb4"]))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_max"]) - (data["flux_d0_pb5"]))) - (((data["flux_d0_pb5"]) + (data["detected_mjd_diff"]))))) - (data["flux_median"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["flux_err_mean"]) + (((((((data["detected_flux_std"]) - (data["detected_mjd_diff"]))) - (data["flux_median"]))) - (data["flux_median"]))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((np.minimum(((data["detected_flux_std"])), ((((data["flux_dif2"]) + (data["detected_flux_mean"])))))) - (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) + (((data["detected_flux_std"]) + (-2.0))))) + (((data["detected_flux_std"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(np.where((-1.0*((data["hostgal_photoz_err"]))) > -1, np.where((2.44751977920532227) > -1, np.where(data["detected_mjd_diff"] > -1, -3.0, 3.0 ), 3.0 ), 3.0 )) +
                0.100000*np.tanh((((-1.0*((((data["flux_d0_pb5"]) + (((data["flux_d0_pb5"]) + (data["flux_d0_pb4"])))))))) - (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, -2.0, (-1.0*((-2.0))) )) +
                0.100000*np.tanh(((((data["detected_flux_std"]) + (((data["flux_err_max"]) + (np.tanh(((-1.0*((data["flux_dif3"])))))))))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) * 2.0)) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_err_mean"]) - (np.where(data["detected_mjd_diff"]<0, data["detected_mjd_diff"], data["flux_err_mean"] )))) - (np.where(data["flux_err_mean"]<0, data["detected_mjd_diff"], (8.0) )))) +
                0.100000*np.tanh((((((data["hostgal_photoz"]) + (data["flux_dif2"]))/2.0)) + (((data["detected_flux_diff"]) + (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"] > -1, -3.0, data["detected_flux_median"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_diff"]) - (data["0__kurtosis_x"]))) - (data["1__kurtosis_x"]))) - (data["0__kurtosis_x"]))) - (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((np.minimum(((data["detected_flux_diff"])), ((data["detected_flux_diff"])))) - (data["1__skewness_x"]))) - (data["1__skewness_x"]))) - ((2.0)))) +
                0.100000*np.tanh(((((((data["detected_flux_diff"]) - (((1.0) - (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))))) - (data["mjd_size"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["detected_flux_std"]) - (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["detected_flux_std"] > -1, data["5__skewness_x"], data["detected_flux_std"] ), ((data["3__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_std"])) )))) +
                0.100000*np.tanh(np.where(((data["hostgal_photoz"]) + (data["flux_d0_pb4"])) > -1, ((data["hostgal_photoz"]) - (3.141593)), (-1.0*((data["hostgal_photoz"]))) )) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_std"])), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] )) + (np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_err_skew"] )))) +
                0.100000*np.tanh(((((((((np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz"])))) * 2.0)) * (((data["detected_mjd_diff"]) + (data["hostgal_photoz"]))))) - (2.718282))) * 2.0)) +
                0.100000*np.tanh(np.where(((data["flux_skew"]) * 2.0) > -1, ((data["hostgal_photoz"]) + (-3.0)), data["detected_flux_std"] )) +
                0.100000*np.tanh(((np.where(data["detected_flux_std"] > -1, ((data["detected_mean"]) - (data["flux_d0_pb5"])), data["flux_w_mean"] )) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["flux_err_mean"]>0, -2.0, ((data["flux_err_mean"]) - (-2.0)) )) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, data["flux_ratio_sq_skew"], ((data["detected_mjd_diff"]) * (-3.0)) )) +
                0.100000*np.tanh(((np.where(data["0__kurtosis_x"]<0, data["4__kurtosis_x"], ((data["hostgal_photoz"]) - (np.where(data["0__kurtosis_x"]<0, data["detected_flux_skew"], data["0__kurtosis_x"] ))) )) + (data["flux_d1_pb3"]))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, ((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((-3.0)), ((data["2__skewness_y"]))))), data["detected_flux_mean"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (data["distmod"])) > -1, ((data["hostgal_photoz"]) * 2.0), ((((data["hostgal_photoz"]) * 2.0)) + (3.0)) )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((-2.0) + (data["hostgal_photoz"])), data["flux_w_mean"] )) +
                0.100000*np.tanh(((((np.minimum(((((((data["detected_flux_std"]) - (data["flux_d1_pb5"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_flux_w_mean"])))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_skew"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__skewness_y"]))))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_std"]) + (((((data["2__skewness_x"]) + (data["2__kurtosis_x"]))) + (data["detected_flux_std"]))))) +
                0.100000*np.tanh((((data["hostgal_photoz"]) + (((((data["hostgal_photoz"]) - (2.0))) + (((data["hostgal_photoz"]) * (((data["hostgal_photoz"]) - ((3.13035678863525391)))))))))/2.0)) +
                0.100000*np.tanh((((((-3.0) + (((data["hostgal_photoz"]) + (data["detected_flux_diff"]))))/2.0)) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((np.where(data["flux_skew"]<0, data["detected_flux_median"], data["flux_max"] )) + (data["flux_dif2"]))) +
                0.100000*np.tanh(((((-1.0) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) - (data["detected_flux_max"]))))) +
                0.100000*np.tanh((-1.0*(((-1.0*(((-1.0*((np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["mjd_diff"], data["detected_flux_err_skew"] ))))))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_max"]>0, np.where(data["flux_d1_pb4"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_0__attr__abs__y"] ), -2.0 )) +
                0.100000*np.tanh(np.where(((((1.0) - (data["flux_d0_pb5"]))) + (data["detected_flux_std"])) > -1, data["detected_flux_std"], data["ddf"] )) +
                0.100000*np.tanh(np.where(data["flux_err_std"]<0, data["detected_flux_std"], ((((data["flux_err_std"]) - (data["flux_err_std"]))) - (((data["detected_flux_std"]) + (data["detected_flux_std"])))) )) +
                0.100000*np.tanh((-1.0*((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + ((((data["flux_d0_pb5"]) > ((-1.0*((data["mjd_size"])))))*1.))))))))) +
                0.100000*np.tanh(((data["flux_d1_pb0"]) + (((((data["flux_w_mean"]) - (data["0__kurtosis_x"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) - (data["distmod"]))) * 2.0)) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) * 2.0)) * (np.where((1.0)<0, ((data["distmod"]) * 2.0), ((data["distmod"]) * 2.0) )))) + (-2.0))) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["flux_d1_pb4"], (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))/2.0) ), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["2__skewness_x"]))))))) +
                0.100000*np.tanh(np.maximum(((data["2__skewness_x"])), ((((data["2__skewness_x"]) / 2.0))))) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, np.where(data["flux_w_mean"] > -1, np.where(data["flux_w_mean"]<0, 2.0, data["detected_flux_max"] ), data["flux_w_mean"] ), -2.0 )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, np.where(data["detected_mjd_diff"]>0, -2.0, -2.0 ), data["4__kurtosis_x"] )) +
                0.100000*np.tanh((((data["detected_flux_ratio_sq_skew"]) + (np.maximum((((((data["detected_flux_w_mean"]) + (data["flux_d1_pb3"]))/2.0))), ((np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))))))))/2.0)) +
                0.100000*np.tanh(np.tanh((((((np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]>0, ((data["detected_flux_std"]) - (data["detected_flux_dif2"])), data["flux_w_mean"] )) * 2.0)) - (data["1__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(np.where(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["2__skewness_x"] )<0, data["hostgal_photoz"], data["flux_by_flux_ratio_sq_skew"] )<0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["mjd_diff"] )) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["detected_flux_err_skew"]))) + (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"] > -1, np.where(np.maximum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_std"]))) > -1, data["detected_flux_skew"], data["detected_flux_err_std"] ), data["flux_dif2"] )) +
                0.100000*np.tanh((((-1.0*((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))) - (((((data["hostgal_photoz_err"]) - ((((data["flux_diff"]) > (data["distmod"]))*1.)))) * (data["3__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_flux_err_mean"], np.where(data["detected_flux_err_max"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["2__kurtosis_x"] ) ) )) +
                0.100000*np.tanh(((((data["detected_flux_dif2"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["ddf"]) - (data["detected_mjd_diff"]))) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["ddf"]))) +
                0.100000*np.tanh((((np.minimum(((data["flux_std"])), ((np.tanh((data["flux_diff"])))))) + (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_d0_pb5"]))))/2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_max"])), ((((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]<0, data["3__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["5__kurtosis_x"]<0, np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["2__skewness_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] ), data["3__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["flux_d1_pb4"] > -1, data["flux_d1_pb4"], np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["flux_d1_pb4"], data["flux_ratio_sq_skew"] ) ), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["detected_mjd_diff"], np.where(data["hostgal_photoz_err"] > -1, data["flux_diff"], data["detected_flux_err_mean"] ) )) +
                0.100000*np.tanh(((data["hostgal_photoz"]) - (3.0))) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["4__kurtosis_x"], data["2__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(np.where(data["2__kurtosis_x"]<0, data["3__skewness_x"], data["2__skewness_x"] ) > -1, data["2__kurtosis_x"], np.where(data["2__kurtosis_x"] > -1, data["2__skewness_x"], data["3__skewness_x"] ) )) +
                0.100000*np.tanh(((((np.minimum(((((data["ddf"]) - (((data["flux_err_max"]) - (data["ddf"])))))), ((data["0__skewness_x"])))) - (data["ddf"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((np.minimum(((-2.0)), ((data["detected_mjd_diff"])))) - (data["distmod"]))) - (((data["distmod"]) * 2.0)))) * 2.0)) - (data["distmod"]))) +
                0.100000*np.tanh((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["hostgal_photoz"]))/2.0)) + (((data["0__kurtosis_y"]) + (((data["detected_flux_dif2"]) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_mjd_diff"]<0, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_mjd_diff"], data["3__kurtosis_y"] ), np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__kurtosis_x"], data["detected_mjd_diff"] ) ))))) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_sum"]<0, ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["mjd_diff"]))) - (data["mjd_diff"]))) / 2.0), ((data["flux_err_std"]) - (data["mjd_diff"])) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] ) ) )) +
                0.100000*np.tanh(np.where(data["distmod"]<0, data["detected_flux_std"], data["detected_flux_err_mean"] )) +
                0.100000*np.tanh((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["hostgal_photoz"]))/2.0)) +
                0.100000*np.tanh(((((data["detected_flux_diff"]) / 2.0)) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.tanh((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["5__kurtosis_y"]<0, (-1.0*((data["2__fft_coefficient__coeff_1__attr__abs__x"]))), np.tanh((np.where(((data["flux_dif2"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) > -1, data["flux_dif2"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ))) )) +
                0.100000*np.tanh(np.where(data["4__kurtosis_x"]>0, data["detected_flux_w_mean"], ((data["detected_flux_dif2"]) / 2.0) )) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], ((data["2__kurtosis_x"]) + (data["detected_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["mjd_size"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))), ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_dif2"])))))) +
                0.100000*np.tanh(((((-1.0*((data["flux_ratio_sq_sum"])))) + ((-1.0*((data["flux_ratio_sq_sum"])))))/2.0)) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["3__fft_coefficient__coeff_1__attr__abs__y"]))>0, np.where(data["flux_d1_pb1"]<0, data["flux_err_mean"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, data["3__fft_coefficient__coeff_0__attr__abs__y"], (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["4__skewness_y"]))/2.0) )) +
                0.100000*np.tanh(((data["hostgal_photoz"]) - (((((((((((((data["distmod"]) * 2.0)) - (data["distmod"]))) * 2.0)) - (data["hostgal_photoz"]))) * 2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((data["distmod"]) * (np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_d1_pb4"], data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) * 2.0)) * 2.0)))

    def GP_class_65(self,data):
        return (-0.972955 +
                0.100000*np.tanh((((-1.0*((((data["distmod"]) + (((data["distmod"]) + (((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) + (2.0)))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["detected_mjd_diff"])), ((((np.minimum(((data["flux_ratio_sq_skew"])), ((((np.minimum(((data["detected_mjd_diff"])), ((data["flux_ratio_sq_skew"])))) * 2.0))))) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -3.0, np.where(data["distmod"] > -1, data["distmod"], ((data["flux_by_flux_ratio_sq_skew"]) * 2.0) ) )) +
                0.100000*np.tanh(((((-2.0) + (data["flux_ratio_sq_skew"]))) + (((((data["detected_mjd_size"]) - (data["detected_mjd_size"]))) - (data["distmod"]))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)))) + (((np.minimum(((data["detected_mjd_diff"])), ((data["flux_by_flux_ratio_sq_skew"])))) - ((2.0)))))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["detected_mjd_diff"], ((((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_ratio_sq_skew"]))))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh(((((((data["flux_by_flux_ratio_sq_skew"]) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((((data["detected_mjd_diff"]) * 2.0))), ((((((((((data["flux_ratio_sq_skew"]) - (data["detected_mjd_size"]))) * 2.0)) * 2.0)) - (data["detected_mjd_size"])))))) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz"] > -1, data["flux_ratio_sq_skew"], data["hostgal_photoz"] ) > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, -2.0, data["hostgal_photoz"] ), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.minimum(((np.where(((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))<0, -3.0, data["detected_mjd_diff"] ))), ((data["detected_mjd_diff"])))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - ((((((-3.0) < (data["4__fft_coefficient__coeff_1__attr__abs__y"]))*1.)) * 2.0)))) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.minimum((((((((data["flux_by_flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) + (-3.0))/2.0))), ((data["flux_by_flux_ratio_sq_skew"])))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["detected_mjd_diff"]))) - (2.0))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_ratio_sq_skew"])), ((((data["detected_flux_err_mean"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) + (np.minimum(((data["flux_ratio_sq_skew"])), ((data["2__kurtosis_x"])))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) + (((((data["flux_by_flux_ratio_sq_skew"]) * (data["detected_mjd_diff"]))) + (((-2.0) - (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) - (((((((data["distmod"]) + ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))/2.0)) + (3.141593))/2.0)))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_skew"]) + (((((((-2.0) - (data["distmod"]))) - (data["distmod"]))) + (((-3.0) + (data["flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, np.where(data["flux_by_flux_ratio_sq_skew"] > -1, data["hostgal_photoz"], np.where(data["hostgal_photoz"] > -1, -2.0, data["flux_by_flux_ratio_sq_skew"] ) ), data["flux_by_flux_ratio_sq_skew"] )) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((data["flux_by_flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["detected_mjd_diff"])))) * 2.0)) + (np.tanh((((data["flux_ratio_sq_skew"]) * (data["detected_mjd_diff"]))))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) - (1.0))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["2__kurtosis_y"] )) - (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((-2.0) - (data["flux_dif3"])), np.where(data["hostgal_photoz"] > -1, ((data["hostgal_photoz"]) - (data["hostgal_photoz"])), data["flux_by_flux_ratio_sq_skew"] ) )) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) + (data["1__skewness_x"]))/2.0)) + ((((data["flux_ratio_sq_skew"]) + (((data["flux_skew"]) * 2.0)))/2.0)))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, np.where(-3.0 > -1, data["hostgal_photoz"], -3.0 ), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, -1.0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) + (-3.0))) + (np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((((np.minimum(((data["2__kurtosis_y"])), ((((-2.0) + (data["detected_mjd_diff"])))))) + (data["detected_mjd_diff"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["4__kurtosis_x"]) + (((data["flux_skew"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((data["flux_skew"]) - (data["distmod"]))) > (data["detected_flux_by_flux_ratio_sq_skew"]))*1.)) - (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(np.where(-3.0<0, np.where(data["distmod"] > -1, -3.0, data["detected_mjd_diff"] ), np.where(data["detected_mjd_diff"] > -1, data["distmod"], data["flux_ratio_sq_skew"] ) )) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["detected_mjd_diff"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) - (np.maximum(((data["2__skewness_y"])), (((((((data["2__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) / 2.0))))))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) - (((data["distmod"]) - (((-2.0) - (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__skewness_x"]))) * 2.0)))) +
                0.100000*np.tanh(np.where(data["flux_skew"]>0, ((data["ddf"]) + (data["2__skewness_x"])), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, (((6.0)) - ((7.0))), (6.0) )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((((np.minimum(((-1.0)), ((-1.0)))) + (data["detected_mjd_diff"]))) + (np.minimum(((0.0)), ((data["flux_by_flux_ratio_sq_sum"])))))))) +
                0.100000*np.tanh(np.where(np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))>0, np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"]<0, np.where(data["hostgal_photoz"] > -1, np.where(data["flux_skew"] > -1, -3.0, data["flux_skew"] ), data["flux_skew"] ), data["flux_skew"] )) +
                0.100000*np.tanh((((-1.0*((data["distmod"])))) + (np.where(data["flux_median"]>0, -3.0, (((-3.0) + (data["3__kurtosis_x"]))/2.0) )))) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) - (((data["flux_d0_pb3"]) - (data["flux_skew"]))))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(((data["flux_by_flux_ratio_sq_skew"]) - (-3.0)) > -1, np.where(((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])) > -1, -3.0, data["flux_ratio_sq_skew"] ), data["flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((((-2.0) + (data["distmod"]))) - (((data["distmod"]) - (-2.0))))) - (data["distmod"]))) - (data["distmod"]))) +
                0.100000*np.tanh(((data["1__kurtosis_y"]) + (((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) + (-2.0))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((np.tanh((-2.0))) + (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((((data["2__skewness_x"]) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__skewness_x"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * (data["flux_skew"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) - ((((2.718282) > (data["detected_flux_ratio_sq_skew"]))*1.)))) +
                0.100000*np.tanh(((data["flux_std"]) + (np.where(data["3__kurtosis_x"]>0, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], (((data["3__kurtosis_x"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))/2.0) )) +
                0.100000*np.tanh((((((data["detected_flux_min"]) + (data["flux_ratio_sq_skew"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(-2.0 > -1, ((((-2.0) - (data["distmod"]))) * 2.0), ((((-2.0) - (data["distmod"]))) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_err_std"] > -1, ((((data["flux_err_std"]) * 2.0)) * 2.0), data["3__skewness_x"] )) * 2.0)) +
                0.100000*np.tanh(((data["2__kurtosis_y"]) + (((data["2__kurtosis_y"]) + (((data["3__skewness_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_median"]) - (data["flux_median"]))))) +
                0.100000*np.tanh(((((-2.0) + (((((np.minimum(((-2.0)), ((-2.0)))) - (data["distmod"]))) - (data["distmod"]))))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["detected_flux_min"]) - (data["flux_mean"]))) * 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["distmod"]))) - (2.0))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["5__skewness_x"]) - (((data["distmod"]) + (3.0))))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["flux_by_flux_ratio_sq_skew"]) - (data["flux_median"]))) - (data["flux_median"]))) - (data["flux_std"]))) - (data["flux_median"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["flux_skew"] ), np.where(data["hostgal_photoz"]>0, data["hostgal_photoz"], -3.0 ) )) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, -3.0, ((data["distmod"]) + ((5.06640815734863281))) )) +
                0.100000*np.tanh(((((((((data["2__skewness_x"]) + (data["0__skewness_y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["flux_d0_pb1"]))) + (data["2__skewness_x"]))) +
                0.100000*np.tanh((((((((data["detected_flux_min"]) + (data["3__kurtosis_y"]))) + (data["5__kurtosis_y"]))) + (data["flux_skew"]))/2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], np.minimum(((data["flux_skew"])), ((data["0__skewness_x"]))) )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((((((data["flux_err_mean"]) * 2.0)) - (data["mwebv"]))) < (((((data["flux_err_std"]) / 2.0)) * (data["2__skewness_y"]))))*1.)))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"] > -1, data["3__kurtosis_x"], data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["flux_ratio_sq_skew"], np.minimum(((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"])))), ((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"]))))) )) +
                0.100000*np.tanh((((np.where((((data["1__kurtosis_y"]) < ((((data["3__kurtosis_x"]) + (data["0__skewness_x"]))/2.0)))*1.)<0, data["detected_flux_ratio_sq_sum"], data["1__kurtosis_x"] )) + (data["1__kurtosis_x"]))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_std"]))))) +
                0.100000*np.tanh(((np.where(((((((((data["flux_err_median"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)>0, data["3__kurtosis_x"], ((data["ddf"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(1.0<0, ((data["distmod"]) - (((-3.0) - (data["distmod"])))), ((-3.0) - (((data["distmod"]) * 2.0))) )) +
                0.100000*np.tanh(((((np.minimum(((-2.0)), ((((-2.0) - (data["distmod"])))))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((data["flux_skew"]) + (np.where(data["flux_skew"]<0, data["flux_err_std"], (((data["1__skewness_x"]) + (np.where(data["0__kurtosis_y"] > -1, data["flux_d0_pb1"], data["flux_err_std"] )))/2.0) )))) +
                0.100000*np.tanh(((((((data["flux_by_flux_ratio_sq_skew"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) - (np.minimum((((-1.0*((data["3__skewness_x"]))))), ((((data["4__kurtosis_y"]) * 2.0))))))) +
                0.100000*np.tanh(((((((((data["0__skewness_x"]) + (data["0__skewness_x"]))/2.0)) + (data["mwebv"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["detected_flux_min"]) * 2.0))), ((((((data["flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (((data["flux_mean"]) - (data["detected_mjd_diff"]))))) - (data["detected_flux_err_skew"]))) - (1.0))) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["flux_diff"] > -1, data["4__kurtosis_x"], data["flux_err_mean"] )) +
                0.100000*np.tanh(((((((data["flux_median"]) - (data["flux_median"]))) - (np.maximum(((-2.0)), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) - (data["flux_std"]))) +
                0.100000*np.tanh(((((((((np.where(((data["distmod"]) / 2.0) > -1, -1.0, 0.367879 )) / 2.0)) * 2.0)) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((((((((((data["flux_d0_pb1"]) - (np.tanh(((7.0)))))) < (data["flux_w_mean"]))*1.)) + (data["flux_std"]))) - (data["flux_d0_pb1"]))) +
                0.100000*np.tanh(((data["0__skewness_x"]) + ((((((data["3__skewness_x"]) + ((((data["3__kurtosis_x"]) + (data["detected_flux_skew"]))/2.0)))/2.0)) + (-2.0))))) +
                0.100000*np.tanh(((((np.where(0.367879 > -1, ((data["ddf"]) + (((((data["flux_err_std"]) * 2.0)) * 2.0))), data["flux_err_std"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_skew"]))))) +
                0.100000*np.tanh(((((data["flux_max"]) + (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((data["1__kurtosis_y"]) + (np.minimum(((((data["flux_ratio_sq_sum"]) + (np.where(data["2__kurtosis_y"] > -1, data["2__kurtosis_y"], data["3__kurtosis_x"] ))))), ((data["1__fft_coefficient__coeff_0__attr__abs__y"])))))/2.0)) +
                0.100000*np.tanh(((data["flux_ratio_sq_sum"]) * (np.tanh((data["detected_flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((np.where(data["flux_err_std"] > -1, ((data["flux_err_std"]) * 2.0), data["flux_err_std"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["detected_flux_skew"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.maximum(((-3.0)), ((data["3__kurtosis_x"]))), data["hostgal_photoz"] )) +
                0.100000*np.tanh(np.maximum(((data["0__skewness_y"])), ((data["1__kurtosis_x"])))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) / 2.0)) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((data["2__kurtosis_x"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (((data["2__skewness_x"]) - (((data["detected_flux_by_flux_ratio_sq_skew"]) - (data["2__skewness_x"]))))))/2.0)) +
                0.100000*np.tanh(((((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)) - (data["flux_d0_pb1"]))) * 2.0)) + (data["flux_skew"]))) +
                0.100000*np.tanh(((((data["flux_err_std"]) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((data["flux_d1_pb5"]) > ((((data["flux_ratio_sq_sum"]) + (data["detected_mjd_diff"]))/2.0)))*1.)) +
                0.100000*np.tanh((((data["flux_err_std"]) + (data["flux_d1_pb5"]))/2.0)) +
                0.100000*np.tanh(((-2.0) + (np.maximum(((np.maximum(((-2.0)), ((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))))))), ((data["detected_mjd_diff"])))))) +
                0.100000*np.tanh(((np.where(data["2__skewness_x"]<0, data["flux_err_std"], ((((((data["2__skewness_x"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["detected_flux_w_mean"]))/2.0) )) * 2.0)) +
                0.100000*np.tanh((((((((data["0__skewness_x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) + (data["0__skewness_x"]))/2.0)) +
                0.100000*np.tanh((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) < (data["detected_flux_min"]))*1.)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_ratio_sq_sum"] )) / 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_sum"] > -1, data["1__skewness_x"], data["3__kurtosis_x"] )) +
                0.100000*np.tanh((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["flux_skew"]) + (data["flux_ratio_sq_skew"]))))/2.0)))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]>0, data["5__kurtosis_x"], np.where(np.minimum(((data["1__skewness_y"])), ((data["flux_err_std"])))>0, np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["5__kurtosis_x"]))), data["flux_err_std"] ) )) +
                0.100000*np.tanh(np.minimum(((np.minimum((((((data["detected_mjd_diff"]) + (np.minimum(((data["flux_mean"])), ((data["0__fft_coefficient__coeff_1__attr__abs__y"])))))/2.0))), ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))))), ((data["3__kurtosis_x"])))) +
                0.100000*np.tanh((((np.minimum(((((data["flux_ratio_sq_skew"]) / 2.0))), ((data["1__skewness_y"])))) + (data["1__skewness_y"]))/2.0)) +
                0.100000*np.tanh(np.maximum(((data["flux_dif3"])), ((data["detected_flux_ratio_sq_sum"])))) +
                0.100000*np.tanh(np.where(data["flux_max"] > -1, -3.0, data["2__skewness_x"] )) +
                0.100000*np.tanh(((((np.where(((-2.0) - (data["distmod"]))<0, -2.0, data["ddf"] )) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["0__skewness_y"], 3.0 )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb3"]>0, ((data["4__kurtosis_x"]) * 2.0), ((np.where(data["flux_d1_pb3"]>0, ((data["detected_flux_median"]) + (data["ddf"])), data["flux_d1_pb3"] )) / 2.0) )))

    def GP_class_67(self,data):
        return (-1.801807 +
                0.100000*np.tanh(((data["4__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (np.minimum(((data["5__kurtosis_x"])), ((data["4__kurtosis_x"])))))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((data["3__skewness_x"]) + (((data["5__kurtosis_x"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((((((data["4__kurtosis_x"]) + (data["detected_flux_min"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["4__skewness_x"]))) + (((data["detected_flux_min"]) + (data["3__kurtosis_x"]))))) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"]))) + (((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["4__kurtosis_x"]) + (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["4__kurtosis_x"]))) + (((((data["3__kurtosis_x"]) + (data["4__kurtosis_x"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(((((((((data["3__kurtosis_x"]) + (data["distmod"]))) + (data["3__kurtosis_x"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) + ((((((data["flux_by_flux_ratio_sq_skew"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))/2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((((data["5__kurtosis_x"]) - (0.0))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["distmod"])))) + (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(np.minimum(((data["3__kurtosis_x"])), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh(((((((data["detected_flux_w_mean"]) + (data["hostgal_photoz_err"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((((data["hostgal_photoz_err"]) + (data["flux_dif2"]))) + (data["4__kurtosis_x"]))/2.0)) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((np.tanh((((data["flux_ratio_sq_skew"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.minimum(((data["flux_dif2"])), ((data["detected_flux_min"]))))), ((data["2__kurtosis_x"]))))))))))), ((data["flux_ratio_sq_skew"])))) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) - (data["0__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((((data["2__kurtosis_y"]) + (((data["2__skewness_x"]) + (np.minimum(((data["ddf"])), ((data["4__kurtosis_x"]))))))))), ((np.minimum(((data["distmod"])), ((data["distmod"]))))))) +
                0.100000*np.tanh((((((data["distmod"]) + (((((data["distmod"]) * 2.0)) + (((data["mjd_size"]) - (data["detected_mjd_size"]))))))/2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["distmod"])), ((np.minimum(((data["distmod"])), ((data["distmod"])))))))), ((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["5__kurtosis_x"])))) +
                0.100000*np.tanh(((((((((data["5__kurtosis_x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_d1_pb0"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_dif2"])), ((data["flux_dif2"])))) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["2__kurtosis_x"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["flux_dif2"]) + (data["3__skewness_y"]))/2.0)) + (np.minimum(((data["0__kurtosis_y"])), ((data["3__skewness_y"])))))/2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((np.minimum(((((data["2__kurtosis_y"]) + (data["flux_d1_pb5"])))), ((data["2__kurtosis_y"]))))))) / 2.0)) +
                0.100000*np.tanh(np.minimum(((((((data["flux_by_flux_ratio_sq_skew"]) - (data["detected_mjd_diff"]))) * 2.0))), ((data["4__skewness_x"])))) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (((np.where(data["detected_mjd_diff"] > -1, ((((data["detected_mjd_diff"]) - (data["3__skewness_x"]))) * 2.0), data["3__skewness_x"] )) * 2.0)))) +
                0.100000*np.tanh((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["distmod"]))/2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((np.tanh((data["4__skewness_x"]))) - (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__skewness_y"]))))) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["3__kurtosis_x"])))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif3"] )) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.minimum(((data["hostgal_photoz_err"])), ((data["1__skewness_y"])))) + (data["detected_flux_std"]))) + (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["5__kurtosis_x"]) - (data["detected_mjd_diff"]))) - ((((data["0__kurtosis_x"]) + (data["detected_mjd_diff"]))/2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d1_pb5"])), ((np.minimum(((data["mjd_size"])), ((data["2__kurtosis_x"]))))))) + (((data["2__kurtosis_y"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((data["detected_flux_std"]) * ((((14.21325874328613281)) * (((data["flux_dif2"]) * ((7.0)))))))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb5"] > -1, ((data["detected_flux_dif3"]) - (data["0__skewness_x"])), ((np.where(data["3__kurtosis_x"]>0, data["flux_d1_pb5"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["flux_d1_pb5"])) )) +
                0.100000*np.tanh(((data["mjd_size"]) + (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) + (data["detected_flux_diff"]))) +
                0.100000*np.tanh((((((data["ddf"]) + (data["distmod"]))/2.0)) + (((data["2__kurtosis_y"]) + (((data["detected_flux_dif3"]) + (data["4__kurtosis_y"]))))))) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + ((((7.57827568054199219)) - (data["detected_flux_std"]))))) * (((((((data["detected_flux_std"]) - (data["flux_d0_pb1"]))) * 2.0)) * 2.0)))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["flux_dif2"]))) + (data["flux_dif2"]))) +
                0.100000*np.tanh(np.minimum(((-1.0)), ((np.where(data["3__skewness_y"]>0, data["3__skewness_y"], data["3__skewness_y"] ))))) +
                0.100000*np.tanh(((np.maximum(((data["flux_dif2"])), (((14.82015228271484375))))) * ((((((14.82015228271484375)) * ((14.82015228271484375)))) * (data["flux_dif2"]))))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["detected_flux_std"]))) * ((((8.74618148803710938)) + (data["detected_flux_std"]))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((((data["flux_d0_pb5"]) + (data["flux_d0_pb5"]))) * 2.0))))) +
                0.100000*np.tanh(((((((((data["flux_max"]) - (data["detected_mjd_diff"]))) - (data["detected_flux_err_median"]))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_std"]) * (((data["detected_flux_std"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["2__skewness_y"]))) +
                0.100000*np.tanh(np.where(((data["detected_flux_dif2"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))>0, ((np.where(data["detected_mean"]>0, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_std"] )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["1__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.maximum(((((data["detected_flux_dif2"]) + (data["ddf"])))), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (((data["detected_mjd_diff"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["flux_dif2"]<0, np.where(data["detected_flux_std"]<0, data["detected_flux_err_skew"], data["flux_max"] ), data["detected_flux_std"] )<0, data["0__kurtosis_x"], data["detected_flux_dif2"] )) +
                0.100000*np.tanh(((((((data["flux_d1_pb5"]) * (data["flux_d0_pb3"]))) - (np.where(data["0__skewness_x"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb0"] )))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]<0, data["5__fft_coefficient__coeff_0__attr__abs__y"], np.minimum(((data["distmod"])), ((data["5__fft_coefficient__coeff_0__attr__abs__y"]))) )) - (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(np.where((((data["flux_max"]) > (data["1__fft_coefficient__coeff_0__attr__abs__x"]))*1.)>0, ((data["distmod"]) + (data["detected_flux_std"])), -2.0 )) +
                0.100000*np.tanh(((((data["detected_flux_dif2"]) - (np.where(data["detected_flux_dif2"] > -1, data["detected_mjd_diff"], (((data["detected_flux_dif2"]) > (data["detected_mjd_diff"]))*1.) )))) - (data["detected_mean"]))) +
                0.100000*np.tanh(((((((data["5__kurtosis_y"]) - (data["flux_by_flux_ratio_sq_sum"]))) + (((((data["1__skewness_y"]) - (data["detected_flux_err_min"]))) + (data["5__kurtosis_y"]))))) - (data["detected_flux_err_median"]))) +
                0.100000*np.tanh(((((((((data["flux_max"]) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["5__skewness_x"]) * (data["2__fft_coefficient__coeff_0__attr__abs__y"])), data["2__skewness_x"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) - (data["flux_d1_pb1"]))<0, -2.0, data["flux_dif2"] )) +
                0.100000*np.tanh(np.where(((data["detected_flux_std"]) - (data["flux_d0_pb2"]))>0, np.where(data["flux_err_std"]>0, data["detected_flux_std"], data["1__kurtosis_x"] ), data["flux_err_std"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.where(data["hostgal_photoz"] > -1, np.tanh((data["detected_flux_std"])), data["hostgal_photoz"] ), ((data["hostgal_photoz_err"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["flux_dif2"]<0, data["flux_dif2"], data["flux_err_skew"] )<0, np.where(data["flux_err_skew"]<0, data["flux_err_skew"], data["flux_dif2"] ), data["ddf"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]<0, np.where(data["flux_d0_pb4"]<0, data["flux_err_skew"], (((data["flux_d0_pb1"]) < (data["flux_std"]))*1.) ), ((data["detected_flux_std"]) * 2.0) )) +
                0.100000*np.tanh(((((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.maximum(((-3.0)), ((((data["0__kurtosis_y"]) * 2.0))))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.where(data["detected_flux_dif2"]<0, data["0__skewness_x"], ((((data["detected_flux_std"]) * (data["3__skewness_y"]))) - (data["0__skewness_x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((((data["0__kurtosis_x"]) + (((np.minimum(((data["flux_err_max"])), ((((data["5__kurtosis_y"]) / 2.0))))) + (data["5__kurtosis_y"])))))))) +
                0.100000*np.tanh(((np.where(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d1_pb5"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ) > -1, data["flux_d1_pb5"], data["2__skewness_y"] )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((np.where(data["flux_d0_pb4"]<0, data["4__fft_coefficient__coeff_0__attr__abs__x"], ((data["detected_flux_std"]) + (np.minimum(((data["distmod"])), ((data["detected_flux_std"]))))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((((np.tanh((data["detected_flux_dif2"]))) > (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["detected_mjd_diff"]>0, data["flux_err_skew"], data["4__fft_coefficient__coeff_1__attr__abs__x"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((np.where(data["2__skewness_y"]>0, data["3__kurtosis_x"], np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_diff"]))) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((((((data["detected_flux_std"]) > (data["flux_d0_pb1"]))*1.)) > (data["flux_d0_pb1"]))*1.), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__skewness_x"]))) + (data["0__kurtosis_x"]))))) * (data["detected_flux_diff"]))) +
                0.100000*np.tanh(((np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], ((data["detected_flux_std"]) * 2.0) )) - (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["hostgal_photoz_err"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"]>0, data["4__skewness_x"], data["flux_min"] ) ) )) +
                0.100000*np.tanh(np.where(np.where((((data["distmod"]) > (data["1__fft_coefficient__coeff_1__attr__abs__y"]))*1.)>0, data["flux_dif2"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )>0, (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) > (data["1__fft_coefficient__coeff_1__attr__abs__y"]))*1.), -3.0 )) +
                0.100000*np.tanh(((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_std"])), ((((data["flux_d0_pb1"]) * (np.minimum(((data["flux_max"])), ((np.tanh((data["detected_flux_diff"]))))))))))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_max"]))*1.)) * 2.0)) - (data["1__fft_coefficient__coeff_0__attr__abs__y"])), data["1__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"]<0, data["0__fft_coefficient__coeff_0__attr__abs__y"], ((((data["1__skewness_y"]) * (data["detected_flux_w_mean"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])) )) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz_err"]<0, np.where(data["detected_mean"] > -1, data["0__skewness_x"], data["hostgal_photoz_err"] ), data["hostgal_photoz_err"] )<0, data["hostgal_photoz_err"], data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((data["detected_flux_dif2"]) + (((((data["4__skewness_y"]) * (data["4__skewness_y"]))) + (((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) * (data["4__skewness_y"]))) + (data["distmod"]))))))) +
                0.100000*np.tanh(((((((((data["detected_flux_diff"]) * (data["1__skewness_y"]))) - (data["flux_d1_pb0"]))) - (data["detected_mjd_diff"]))) - (((data["flux_d1_pb0"]) - (data["detected_flux_mean"]))))) +
                0.100000*np.tanh(((((np.where(data["flux_d0_pb0"]>0, ((data["detected_flux_std"]) - (data["flux_d0_pb0"])), ((data["flux_err_skew"]) - (data["1__skewness_x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) * (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"]<0, data["flux_d0_pb4"], np.where(data["detected_mjd_diff"]>0, np.where(data["flux_d0_pb0"]<0, data["flux_err_min"], data["3__fft_coefficient__coeff_0__attr__abs__x"] ), data["detected_mjd_diff"] ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["4__kurtosis_y"], ((data["flux_d1_pb3"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (np.where(data["5__skewness_x"]>0, data["flux_d1_pb3"], data["4__skewness_x"] ))))) )) +
                0.100000*np.tanh(((((np.where(np.where(data["detected_flux_diff"]<0, data["flux_max"], data["flux_diff"] )<0, data["flux_diff"], data["flux_max"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["0__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["3__kurtosis_x"]<0, data["detected_flux_err_mean"], ((data["distmod"]) - (data["detected_flux_err_median"])) ) )) +
                0.100000*np.tanh((((((data["detected_mjd_diff"]) < (np.tanh((data["detected_flux_dif2"]))))*1.)) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb5"], ((data["3__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_ratio_sq_skew"])) )) +
                0.100000*np.tanh(np.where(np.maximum(((data["2__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) > -1, np.where(data["mjd_diff"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__x"], data["detected_mjd_size"] ), data["2__skewness_y"] )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, (-1.0*((data["4__skewness_y"]))), (((data["3__skewness_x"]) + (data["flux_d1_pb1"]))/2.0) )) +
                0.100000*np.tanh(((data["flux_err_max"]) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__skewness_y"]))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_err_skew"], ((((np.where(data["flux_std"]<0, data["hostgal_photoz_err"], data["flux_err_max"] )) * 2.0)) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (((data["detected_flux_std"]) + (data["distmod"]))))<0, data["distmod"], ((data["detected_flux_std"]) - (data["distmod"])) )) +
                0.100000*np.tanh(np.where((((data["detected_mjd_diff"]) < (np.tanh(((((data["detected_flux_dif2"]) + (data["detected_mjd_diff"]))/2.0)))))*1.)>0, data["detected_flux_dif2"], -3.0 )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, ((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d1_pb5"])))) )) +
                0.100000*np.tanh(((((((((data["5__kurtosis_y"]) + (((data["flux_d1_pb5"]) + (data["distmod"]))))) + (data["flux_median"]))) + (data["1__kurtosis_y"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["5__skewness_y"]<0, ((((data["5__kurtosis_y"]) + (data["distmod"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["1__skewness_y"] )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"]<0, data["mjd_diff"], ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where((((data["detected_flux_w_mean"]) < (np.where(data["flux_mean"]<0, data["flux_d1_pb4"], data["flux_mean"] )))*1.)>0, (-1.0*((data["flux_d1_pb4"]))), data["flux_d1_pb4"] )) +
                0.100000*np.tanh(((((data["flux_skew"]) - (data["3__kurtosis_y"]))) - (np.maximum(((data["flux_d1_pb1"])), ((data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((data["4__skewness_y"]) * (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["flux_d0_pb4"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))))) +
                0.100000*np.tanh(((np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["2__kurtosis_x"])), data["3__kurtosis_x"] ), data["3__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_std"]>0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__kurtosis_x"], data["flux_std"] ), data["detected_flux_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_std"]<0, ((((data["hostgal_photoz_err"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["detected_flux_by_flux_ratio_sq_skew"])), np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_by_flux_ratio_sq_skew"], data["4__skewness_y"] ) )) +
                0.100000*np.tanh(((((((((((data["flux_max"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_skew"] > -1, ((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_ratio_sq_skew"]))) * 2.0), 2.0 )) * 2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["distmod"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])), np.tanh((data["3__fft_coefficient__coeff_1__attr__abs__x"])) ), data["detected_flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((data["detected_flux_dif2"]) - (data["detected_flux_err_median"]))) * 2.0)) + (data["4__kurtosis_y"]))) +
                0.100000*np.tanh(np.where(np.where(((data["flux_err_std"]) + ((((data["1__skewness_y"]) + (data["flux_err_std"]))/2.0)))>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )>0, data["flux_err_std"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) * 2.0) > -1, (((((-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)) * 2.0), ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) * (np.where(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * (data["detected_flux_err_skew"]))) + (data["detected_flux_err_skew"]))>0, data["detected_flux_err_skew"], data["2__fft_coefficient__coeff_1__attr__abs__y"] )))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, data["detected_flux_diff"], np.where(data["flux_median"]>0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["flux_median"] > -1, data["flux_median"], data["flux_median"] ) ) )))

    def GP_class_88(self,data):
        return (-1.503109 +
                0.100000*np.tanh(((((data["distmod"]) + (((((data["distmod"]) * 2.0)) * 2.0)))) + (np.where(data["distmod"] > -1, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["distmod"] )))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]>0, data["distmod"], data["distmod"] )) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["distmod"]) + (((data["distmod"]) - (data["3__kurtosis_x"]))))))) +
                0.100000*np.tanh((((((np.minimum(((data["distmod"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (((data["distmod"]) + (((data["distmod"]) - (data["2__kurtosis_x"]))))))/2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["distmod"]) + ((((((data["distmod"]) > (data["distmod"]))*1.)) - (data["4__kurtosis_x"]))))) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))) + (data["distmod"]))) + (((data["distmod"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(((data["distmod"]) - (data["distmod"])) > -1, np.minimum(((data["distmod"])), ((data["distmod"]))), ((data["0__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0) )) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((np.where(data["2__skewness_x"]>0, -3.0, ((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"])) )) * 2.0)))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"])))) * 2.0))), ((((data["distmod"]) - (data["2__kurtosis_x"])))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_mean"]) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(((((np.where(data["distmod"] > -1, data["detected_mjd_diff"], -1.0 )) + (-1.0))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["flux_skew"]) - (data["flux_skew"]))) - (data["flux_skew"]))) + (data["detected_mjd_diff"]))) - (((data["flux_skew"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) + (((((((data["distmod"]) + ((-1.0*((data["4__skewness_x"])))))) * 2.0)) * 2.0)))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((((data["distmod"]) - (data["2__skewness_x"]))) - (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["4__kurtosis_x"]))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, ((data["3__kurtosis_x"]) - (((((((data["3__skewness_x"]) * 2.0)) * 2.0)) * 2.0))), data["hostgal_photoz"] )) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((((data["distmod"]) * 2.0)) + (data["distmod"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["distmod"]) - (data["flux_skew"]))) - (data["1__kurtosis_x"]))) - (data["flux_skew"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, np.where(data["2__skewness_x"] > -1, data["distmod"], ((data["distmod"]) + (data["distmod"])) ), -3.0 )) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]<0, np.where(data["distmod"]<0, np.where(-3.0<0, data["distmod"], data["3__skewness_x"] ), data["detected_mjd_diff"] ), -2.0 )) +
                0.100000*np.tanh(((np.where(np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"]))) > -1, data["distmod"], data["distmod"] )) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["hostgal_photoz"] > -1, data["detected_mjd_diff"], data["flux_skew"] )) - (data["detected_flux_median"]))) - (data["3__skewness_x"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh((((((((-1.0*((data["flux_skew"])))) * 2.0)) - (np.where(data["flux_ratio_sq_sum"] > -1, data["3__skewness_x"], ((data["flux_d1_pb2"]) * (data["detected_mjd_size"])) )))) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((np.where(data["1__kurtosis_x"]>0, -2.0, np.where(data["1__kurtosis_x"]<0, ((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0), data["1__fft_coefficient__coeff_1__attr__abs__y"] ) )) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((data["distmod"])), ((data["distmod"]))))))) * 2.0)))) +
                0.100000*np.tanh(np.where(data["flux_skew"]>0, data["hostgal_photoz"], ((data["distmod"]) * 2.0) )) +
                0.100000*np.tanh(((((((data["4__skewness_x"]) * (data["detected_mean"]))) / 2.0)) - ((((12.28340435028076172)) * (data["4__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - (data["detected_flux_diff"]))) - (np.maximum(((data["detected_flux_std"])), ((data["1__skewness_x"])))))) - (data["flux_skew"]))) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"])))) * 2.0))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0)) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mean"]) - (data["4__skewness_x"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (((data["detected_mjd_diff"]) - (0.367879))))*1.)))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) * 2.0))), ((data["distmod"])))) + (((((data["distmod"]) + (data["detected_mjd_diff"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]>0, np.where(data["flux_skew"]>0, -3.0, data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.minimum(((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * 2.0)) * 2.0)) * 2.0))), ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["detected_mjd_diff"]) - (data["distmod"])), ((data["distmod"]) + (data["distmod"])) )) +
                0.100000*np.tanh(((((data["distmod"]) + (data["distmod"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.minimum(((((data["distmod"]) - (data["1__skewness_x"])))), ((np.where(data["detected_mjd_diff"]>0, data["detected_mjd_diff"], data["flux_skew"] ))))) - (data["flux_d1_pb2"]))) * 2.0)) +
                0.100000*np.tanh(((((((((-1.0) + (data["detected_mjd_diff"]))) + (data["distmod"]))) * 2.0)) + (((-1.0) * 2.0)))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_err_min"]) + (data["distmod"]))) * 2.0)) + (data["detected_mjd_diff"]))) + (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((np.tanh((((np.tanh((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)))) * 2.0)))) * 2.0))))) +
                0.100000*np.tanh(((((data["detected_mean"]) + (data["distmod"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) - (data["detected_flux_by_flux_ratio_sq_sum"]))) - (data["flux_skew"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))*1.)))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (((data["distmod"]) * 2.0)))) + (((data["distmod"]) + (-3.0))))) +
                0.100000*np.tanh(((((((((data["detected_mean"]) - (data["flux_d0_pb3"]))) - (data["1__kurtosis_x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + ((((((((data["distmod"]) + (data["distmod"]))) + (np.minimum(((data["hostgal_photoz"])), ((data["distmod"])))))/2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((((-1.0) + (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) + (-1.0))) +
                0.100000*np.tanh(np.where(np.minimum(((data["detected_mjd_diff"])), ((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_ratio_sq_skew"])))), ((data["detected_flux_ratio_sq_skew"]))))))>0, data["detected_mjd_diff"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((np.where(data["flux_skew"] > -1, data["detected_mjd_diff"], data["flux_err_min"] )) > (((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (((((data["detected_mjd_diff"]) + (((data["flux_err_min"]) + (data["detected_mjd_diff"]))))) + (data["flux_err_min"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_skew"]))) +
                0.100000*np.tanh(((data["distmod"]) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((((((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["3__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["detected_mjd_diff"])))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["distmod"]) + (((data["0__skewness_x"]) - ((((data["0__skewness_x"]) > (data["detected_mjd_diff"]))*1.)))))) + (data["distmod"]))) +
                0.100000*np.tanh(((np.where(data["flux_skew"]>0, -2.0, ((((data["flux_err_min"]) * 2.0)) * 2.0) )) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, data["detected_mjd_diff"], ((data["distmod"]) + (np.where(data["distmod"] > -1, data["distmod"], np.minimum(((data["flux_by_flux_ratio_sq_sum"])), ((data["flux_d0_pb5"]))) ))) )) +
                0.100000*np.tanh(((((np.where(((data["flux_err_median"]) * 2.0) > -1, ((data["detected_mean"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"])), np.minimum(((data["detected_mean"])), ((data["detected_mean"]))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["distmod"]) + ((((data["detected_mjd_diff"]) + (((((data["flux_d0_pb0"]) + (data["detected_mjd_diff"]))) + (data["distmod"]))))/2.0)))) + (data["distmod"]))) +
                0.100000*np.tanh(np.where((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_by_flux_ratio_sq_sum"]))/2.0) > -1, ((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_mjd_diff"])) )) +
                0.100000*np.tanh(((((data["detected_flux_dif3"]) + (((((((data["flux_err_min"]) + (data["detected_mean"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)))) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["flux_err_min"], (-1.0*((data["1__fft_coefficient__coeff_1__attr__abs__y"]))) )) + (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"] > -1, ((((((((data["flux_median"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["flux_d0_pb0"]))) * 2.0), data["flux_d0_pb2"] )) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((((((data["distmod"]) + (data["distmod"]))) + (data["detected_flux_ratio_sq_skew"]))) + (data["distmod"])))))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], ((data["detected_mjd_diff"]) + (np.tanh((((data["flux_err_min"]) * 2.0))))) )) +
                0.100000*np.tanh(np.where(np.where(data["detected_flux_err_skew"]<0, data["detected_flux_diff"], data["detected_mjd_diff"] ) > -1, data["detected_mjd_diff"], data["detected_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.minimum(((((((np.minimum(((data["detected_mjd_diff"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)) * 2.0))), ((((data["detected_mean"]) * 2.0))))) +
                0.100000*np.tanh(np.minimum(((((((((data["flux_median"]) - (data["detected_flux_max"]))) - (data["flux_std"]))) - (data["1__kurtosis_x"])))), ((data["flux_median"])))) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) + (((np.where(data["detected_mjd_diff"] > -1, ((-2.0) + (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) * 2.0)))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (((data["hostgal_photoz"]) + (((data["flux_err_min"]) + (data["distmod"]))))))) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["distmod"]))) +
                0.100000*np.tanh((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((np.where(data["detected_mean"]<0, ((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"])), data["0__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)))/2.0)) +
                0.100000*np.tanh(((np.where(data["flux_err_max"] > -1, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["distmod"] )) + (np.minimum(((data["hostgal_photoz"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_err_min"])), ((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (data["distmod"]))))))), ((data["distmod"])))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (((data["detected_mjd_diff"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_err_min"]))) +
                0.100000*np.tanh(((((np.where(data["0__skewness_y"]>0, data["detected_flux_ratio_sq_skew"], np.where(data["flux_err_min"] > -1, data["flux_err_min"], data["detected_flux_ratio_sq_skew"] ) )) * 2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["flux_dif3"]))) + (((data["4__skewness_y"]) + (((((data["detected_mjd_diff"]) - (data["detected_flux_mean"]))) / 2.0)))))) +
                0.100000*np.tanh(np.where(np.maximum(((data["flux_d1_pb1"])), ((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (data["detected_flux_diff"]))))) > -1, data["detected_flux_ratio_sq_skew"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_d0_pb1"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], np.where((((-1.0*((data["0__fft_coefficient__coeff_0__attr__abs__x"])))) / 2.0) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d0_pb1"] ) )) +
                0.100000*np.tanh((((data["distmod"]) < (np.minimum(((data["distmod"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))*1.)) +
                0.100000*np.tanh((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((np.minimum(((data["flux_err_min"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (np.minimum(((data["4__skewness_y"])), ((data["detected_mean"])))))))/2.0)) +
                0.100000*np.tanh(((((data["flux_err_min"]) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) + ((((data["distmod"]) + ((((((data["distmod"]) + (data["distmod"]))) + (data["flux_d1_pb4"]))/2.0)))/2.0)))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["2__skewness_y"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) + (data["distmod"]))) * 2.0)) - (np.where(data["mjd_diff"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__y"] )))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb4"] > -1, data["flux_err_min"], ((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) )) +
                0.100000*np.tanh(((((data["flux_err_min"]) * 2.0)) / 2.0)) +
                0.100000*np.tanh((((data["distmod"]) + (data["flux_d0_pb0"]))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_flux_err_mean"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["distmod"]))/2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["detected_flux_diff"]))/2.0)) + (np.maximum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["flux_err_min"]) * 2.0))))))) +
                0.100000*np.tanh(np.tanh((((data["flux_min"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((data["3__skewness_y"])), ((data["3__skewness_y"])))) + (0.367879)))), ((np.minimum(((data["flux_err_min"])), ((((data["5__skewness_y"]) + (data["flux_err_min"]))))))))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (np.where(data["flux_d0_pb3"]<0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["4__skewness_x"])) )))) +
                0.100000*np.tanh(((data["5__kurtosis_y"]) + (np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((np.tanh((data["4__kurtosis_y"]))) + (data["detected_mjd_diff"])))))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__kurtosis_y"]))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["flux_err_min"] > -1, data["detected_flux_ratio_sq_skew"], ((data["detected_flux_ratio_sq_skew"]) * 2.0) )) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_d0_pb0"]))) + ((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_median"]))) < (np.tanh((data["0__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (data["1__skewness_y"]))>0, (((data["detected_flux_ratio_sq_skew"]) > (np.minimum(((data["distmod"])), ((data["5__skewness_y"])))))*1.), data["distmod"] )) +
                0.100000*np.tanh(((((data["3__skewness_y"]) + (data["4__kurtosis_y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((data["flux_err_min"]) - (data["detected_flux_min"])), data["hostgal_photoz"] )) +
                0.100000*np.tanh((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_median"], data["0__fft_coefficient__coeff_1__attr__abs__y"] )) + (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["mjd_size"]<0, data["flux_median"], np.where(data["detected_flux_min"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d1_pb5"]))) + (data["5__kurtosis_x"])) ) )) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_ratio_sq_skew"])), ((data["0__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(((np.where(data["flux_err_std"] > -1, data["flux_median"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["distmod"]) + (np.where(np.tanh((data["detected_flux_min"]))>0, data["detected_flux_err_skew"], data["4__skewness_y"] )))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(3.0 > -1, data["distmod"], ((data["detected_mean"]) - ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.))) )) +
                0.100000*np.tanh((((((((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__skewness_y"]))/2.0)) - (data["1__kurtosis_x"]))) - ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(np.where(data["flux_err_min"]>0, data["flux_err_min"], data["flux_err_min"] )>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_err_min"], data["flux_err_min"] ) )) +
                0.100000*np.tanh(np.minimum(((((data["4__skewness_y"]) + (data["3__skewness_y"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_median"]))) + (data["flux_median"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["detected_mean"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__kurtosis_x"]>0, data["detected_mjd_diff"], np.tanh((data["flux_err_min"])) ) ))), ((0.0)))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (((data["detected_flux_max"]) * 2.0)))) - (data["detected_flux_min"]))) - (data["flux_d0_pb3"]))))

    def GP_class_90(self,data):
        return (-0.436273 +
                0.100000*np.tanh(np.minimum(((-3.0)), ((np.minimum(((np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((3.141593))))), ((-3.0))))))) +
                0.100000*np.tanh(((np.minimum(((-3.0)), ((-2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((-3.0)), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((np.minimum(((np.minimum(((-3.0)), ((data["3__kurtosis_y"]))))), ((-3.0))))), ((-3.0))))))) +
                0.100000*np.tanh((((-3.0) + (np.minimum(((np.minimum(((-3.0)), ((np.minimum(((-3.0)), ((data["4__kurtosis_x"])))))))), ((np.minimum(((data["2__skewness_x"])), ((-3.0))))))))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((data["distmod"])), ((data["distmod"])))>0, data["flux_by_flux_ratio_sq_skew"], -2.0 ))), ((data["distmod"])))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((((((data["distmod"]) * 2.0)) * 2.0))), ((data["3__kurtosis_x"]))))), ((data["distmod"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["flux_ratio_sq_skew"]) + (data["distmod"])))), ((((np.minimum(((data["distmod"])), ((data["distmod"])))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((np.minimum(((data["4__kurtosis_x"])), ((data["distmod"]))))), ((data["distmod"])))) + (data["2__skewness_x"])))), ((np.minimum(((data["distmod"])), ((data["4__kurtosis_x"]))))))) +
                0.100000*np.tanh(np.minimum(((-2.0)), ((-2.0)))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (np.minimum(((data["distmod"])), ((data["4__kurtosis_x"])))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((-3.0) / 2.0))), ((np.minimum(((data["4__kurtosis_x"])), ((data["3__skewness_x"]))))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"]))) * 2.0)) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((data["distmod"]) * 2.0)))))), ((data["3__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(((data["flux_ratio_sq_skew"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_dif2"])), ((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((data["flux_min"]) - (data["hostgal_photoz"]))))))))) - (data["detected_flux_err_max"]))) * 2.0)) +
                0.100000*np.tanh(((((data["distmod"]) + (data["distmod"]))) + (((data["distmod"]) + (data["flux_min"]))))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) * 2.0))), ((np.minimum(((((((np.minimum(((data["distmod"])), ((data["flux_by_flux_ratio_sq_skew"])))) * 2.0)) * 2.0))), ((data["distmod"]))))))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((data["distmod"])))) + (data["flux_min"]))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((data["distmod"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_skew"]) + (np.minimum(((data["2__kurtosis_x"])), ((data["distmod"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["5__skewness_y"])), ((np.minimum(((data["flux_d0_pb3"])), ((data["detected_flux_min"])))))))), (((-1.0*((np.minimum(((data["flux_std"])), ((data["3__kurtosis_x"])))))))))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (((data["4__kurtosis_x"]) + (data["distmod"]))))) * 2.0)) + (((((data["detected_flux_max"]) / 2.0)) + (data["distmod"]))))) +
                0.100000*np.tanh(((np.minimum(((data["3__kurtosis_x"])), ((((data["distmod"]) + (((((data["3__skewness_x"]) + (data["2__kurtosis_x"]))) - (data["detected_flux_err_std"])))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["3__kurtosis_x"]) - (data["detected_flux_err_max"]))) - (0.367879))) - (data["flux_d0_pb3"]))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d1_pb2"]) + (data["4__kurtosis_x"])))), ((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + ((((((((data["4__kurtosis_x"]) * 2.0)) + (data["distmod"]))/2.0)) + (((data["distmod"]) / 2.0)))))) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__kurtosis_x"])), ((data["flux_d0_pb3"]))))), ((np.minimum(((data["3__kurtosis_x"])), ((data["flux_median"]))))))) +
                0.100000*np.tanh(((((((data["detected_flux_min"]) - (data["detected_flux_err_median"]))) * 2.0)) + (((data["4__kurtosis_x"]) + (data["mjd_diff"]))))) +
                0.100000*np.tanh(np.minimum(((((((data["distmod"]) - (data["hostgal_photoz"]))) + (((data["distmod"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])))))), ((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])))))) +
                0.100000*np.tanh((((((((data["flux_min"]) + (((data["flux_min"]) * 2.0)))/2.0)) * 2.0)) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((((((((data["distmod"]) * 2.0)) + (data["flux_d0_pb2"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb2"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - ((((((data["flux_d1_pb2"]) - (data["0__skewness_y"]))) < (data["flux_dif2"]))*1.)))) +
                0.100000*np.tanh((((((((((6.0)) / 2.0)) + (data["flux_min"]))) - (data["4__kurtosis_x"]))) - (data["hostgal_photoz"]))) +
                0.100000*np.tanh((((((-1.0*((data["2__fft_coefficient__coeff_1__attr__abs__x"])))) + (((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["distmod"])), ((np.minimum(((data["distmod"])), ((data["flux_dif2"]))))))) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) + (data["flux_d0_pb1"]))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) / 2.0))), ((np.where(data["distmod"]<0, (((data["distmod"]) + ((-1.0*((data["flux_dif2"])))))/2.0), data["1__kurtosis_x"] ))))) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (data["flux_d0_pb2"]))) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) > -1, ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["flux_by_flux_ratio_sq_skew"]) - (data["hostgal_photoz"])))))), data["flux_median"] )) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (((data["flux_d0_pb2"]) + (((data["distmod"]) * 2.0)))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_dif2"])), ((((data["distmod"]) + (data["distmod"]))))))), ((data["distmod"])))) +
                0.100000*np.tanh((((((data["4__skewness_x"]) + (data["flux_min"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_d0_pb2"]))))/2.0)) +
                0.100000*np.tanh((((data["detected_flux_min"]) + (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_d1_pb1"]))))/2.0)) +
                0.100000*np.tanh(((((((((data["3__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["hostgal_photoz"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.tanh((((((((data["distmod"]) * 2.0)) + (data["flux_d0_pb1"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["1__skewness_x"])), ((((data["5__kurtosis_x"]) * 2.0)))))))) +
                0.100000*np.tanh(((data["1__skewness_y"]) + (data["distmod"]))) +
                0.100000*np.tanh((((((data["5__kurtosis_x"]) > ((((data["hostgal_photoz"]) < (data["hostgal_photoz"]))*1.)))*1.)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["2__kurtosis_x"]>0, data["flux_d0_pb2"], data["2__fft_coefficient__coeff_0__attr__abs__x"] )) * (data["2__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((data["flux_d0_pb2"])), ((data["detected_flux_min"]))) > -1, data["5__kurtosis_x"], data["flux_ratio_sq_skew"] ))), ((data["flux_d0_pb2"])))) +
                0.100000*np.tanh(((np.minimum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_0__attr__abs__x"])))) * (data["2__kurtosis_x"]))) +
                0.100000*np.tanh((((data["flux_d0_pb3"]) + (data["flux_d0_pb3"]))/2.0)) +
                0.100000*np.tanh(np.where(data["distmod"]>0, np.where(data["distmod"] > -1, 2.0, data["distmod"] ), -2.0 )) +
                0.100000*np.tanh(((((((-1.0*((data["4__skewness_x"])))) < (np.tanh((data["hostgal_photoz"]))))*1.)) + ((-1.0*((data["hostgal_photoz"])))))) +
                0.100000*np.tanh((((data["flux_d0_pb2"]) + ((((np.tanh((data["1__skewness_y"]))) + (data["1__skewness_y"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, np.where(2.0<0, data["flux_err_median"], data["detected_mean"] ), ((data["2__kurtosis_y"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh((((data["flux_by_flux_ratio_sq_skew"]) + (np.where(data["distmod"]>0, data["4__kurtosis_x"], data["distmod"] )))/2.0)) +
                0.100000*np.tanh(((np.where(((data["flux_median"]) + (data["flux_median"]))>0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )) * (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(((data["detected_flux_w_mean"]) * (np.where(data["detected_flux_err_std"]>0, data["ddf"], data["flux_d0_pb4"] )))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["detected_flux_min"]))) > (data["flux_d1_pb5"]))*1.)) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["flux_median"] > -1, data["flux_d0_pb1"], data["flux_w_mean"] ), ((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_median"], ((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, np.maximum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))), data["detected_flux_w_mean"] )) + (data["detected_flux_by_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(np.where(((data["detected_flux_err_median"]) * (data["3__fft_coefficient__coeff_0__attr__abs__x"]))<0, (((data["flux_d0_pb3"]) + (data["flux_diff"]))/2.0), (((data["flux_d0_pb3"]) < (data["1__skewness_y"]))*1.) )) +
                0.100000*np.tanh(((np.where(data["flux_median"]>0, data["flux_by_flux_ratio_sq_skew"], ((data["distmod"]) + (((data["distmod"]) + (data["detected_flux_w_mean"])))) )) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb2"]) + (((data["flux_d0_pb2"]) * 2.0)))) + (((data["1__skewness_y"]) + (data["detected_flux_err_min"]))))) +
                0.100000*np.tanh(np.minimum(((((data["5__kurtosis_x"]) / 2.0))), ((((np.tanh((data["flux_by_flux_ratio_sq_skew"]))) * (np.tanh((data["flux_median"])))))))) +
                0.100000*np.tanh(np.where(data["mjd_size"] > -1, (((data["flux_d0_pb5"]) < ((((((data["0__kurtosis_y"]) / 2.0)) > (data["flux_d0_pb5"]))*1.)))*1.), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((((((data["detected_flux_max"]) + (data["distmod"]))) + (data["distmod"]))) * 2.0)) + (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((((data["detected_flux_std"]) + (((data["2__skewness_y"]) + (np.minimum(((data["3__kurtosis_y"])), ((data["flux_d0_pb1"])))))))/2.0)) +
                0.100000*np.tanh((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) > (data["flux_w_mean"]))*1.)) +
                0.100000*np.tanh(((data["flux_dif2"]) * (((data["detected_flux_median"]) - (((data["detected_flux_median"]) * (((data["4__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) * (np.where(data["0__skewness_x"]<0, data["detected_flux_by_flux_ratio_sq_skew"], data["flux_d0_pb4"] )))) * (((data["flux_d1_pb1"]) * 2.0)))) +
                0.100000*np.tanh(np.where((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) > (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.)<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["3__kurtosis_x"])) )) +
                0.100000*np.tanh(((np.where((((data["flux_d0_pb3"]) > (data["flux_d0_pb5"]))*1.)>0, (((data["flux_d0_pb3"]) > (data["flux_dif3"]))*1.), ((data["flux_dif3"]) - (data["flux_d0_pb5"])) )) * 2.0)) +
                0.100000*np.tanh(np.maximum(((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["detected_flux_by_flux_ratio_sq_skew"])))), ((data["flux_d0_pb3"])))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, data["flux_median"], data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((data["mjd_size"]) + (np.maximum(((data["detected_flux_err_mean"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(((((((((data["distmod"]) + (((((data["hostgal_photoz"]) + (data["detected_flux_max"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(np.maximum(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))), ((data["flux_err_skew"]))) > -1, data["5__kurtosis_x"], data["detected_flux_err_median"] )) +
                0.100000*np.tanh(((np.where(data["3__kurtosis_x"] > -1, data["3__kurtosis_x"], ((((data["detected_mjd_size"]) + (data["3__kurtosis_x"]))) * 2.0) )) * (((data["detected_mjd_size"]) * 2.0)))) +
                0.100000*np.tanh(((data["detected_mjd_size"]) * (((((((((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.tanh((np.where(data["flux_median"] > -1, data["flux_median"], data["flux_median"] )))) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, np.minimum(((data["3__kurtosis_x"])), ((data["flux_d0_pb0"]))), np.where(data["detected_flux_min"]>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_dif3"] ) )) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * (np.where(data["mwebv"]<0, data["3__kurtosis_x"], data["3__fft_coefficient__coeff_1__attr__abs__y"] )))) / 2.0)) +
                0.100000*np.tanh(np.where((((data["distmod"]) + (data["flux_err_skew"]))/2.0)>0, data["1__skewness_y"], data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((((((((((((data["detected_flux_max"]) + (data["hostgal_photoz"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["0__skewness_x"]) > (data["hostgal_photoz"]))*1.)) > (data["flux_d1_pb0"]))*1.)) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) * 2.0)) * (((data["3__kurtosis_x"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_median"]) * (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, np.minimum(((data["5__kurtosis_x"])), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))), np.minimum(((data["flux_d0_pb2"])), ((np.minimum(((data["flux_d1_pb2"])), ((data["flux_d0_pb2"])))))) )) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"] > -1, ((data["flux_by_flux_ratio_sq_skew"]) * (data["detected_flux_ratio_sq_skew"])), np.where(data["flux_ratio_sq_sum"] > -1, ((data["flux_by_flux_ratio_sq_skew"]) * (data["flux_ratio_sq_skew"])), data["0__skewness_x"] ) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((data["detected_flux_err_mean"]) * (data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["5__fft_coefficient__coeff_0__attr__abs__x"])) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__skewness_x"]))))) +
                0.100000*np.tanh((((data["flux_median"]) > (data["detected_flux_mean"]))*1.)) +
                0.100000*np.tanh(((((((data["flux_d0_pb2"]) + (((data["flux_median"]) + (data["flux_median"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_median"]))))) + (data["flux_median"]))) +
                0.100000*np.tanh(np.where(data["mjd_diff"] > -1, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["4__skewness_x"], data["detected_flux_min"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_min"]<0, np.where(data["detected_flux_err_min"]>0, data["flux_max"], data["mjd_size"] ), data["mwebv"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, ((((data["distmod"]) * 2.0)) * 2.0), data["5__skewness_x"] )) +
                0.100000*np.tanh(((np.where(((data["3__skewness_x"]) * (data["detected_flux_by_flux_ratio_sq_skew"])) > -1, data["flux_err_skew"], data["detected_flux_by_flux_ratio_sq_skew"] )) + (((data["detected_flux_by_flux_ratio_sq_skew"]) * (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, np.maximum(((data["distmod"])), ((data["3__kurtosis_y"]))), np.tanh((data["flux_d0_pb2"])) )) +
                0.100000*np.tanh((((((((data["flux_d0_pb4"]) / 2.0)) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) * (data["flux_max"]))) +
                0.100000*np.tanh(np.where(np.where(data["flux_by_flux_ratio_sq_skew"]>0, data["flux_d1_pb1"], data["2__skewness_y"] )>0, data["flux_by_flux_ratio_sq_skew"], (((((data["flux_err_max"]) + (((data["2__kurtosis_y"]) / 2.0)))/2.0)) * 2.0) )) +
                0.100000*np.tanh(np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((((data["hostgal_photoz_err"]) * (data["flux_diff"])))), (((((data["flux_diff"]) + (data["flux_diff"]))/2.0))))))))))) +
                0.100000*np.tanh(np.where(data["flux_min"] > -1, np.where(data["flux_d0_pb0"]>0, data["flux_skew"], data["flux_d0_pb0"] ), data["detected_flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"] > -1, np.where((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.)>0, data["mjd_diff"], data["mjd_size"] ), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]>0, data["flux_median"], data["detected_flux_err_std"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh((((np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif2"] )) > (data["detected_mjd_diff"]))*1.)) +
                0.100000*np.tanh(np.where(((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"])) > -1, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], -2.0 ), -2.0 )) +
                0.100000*np.tanh(np.where(((data["flux_diff"]) + (((data["distmod"]) + (data["distmod"]))))<0, ((data["distmod"]) + (data["hostgal_photoz_err"])), 2.718282 )) +
                0.100000*np.tanh(((data["4__skewness_x"]) * (((np.where(data["3__kurtosis_x"]>0, data["3__kurtosis_x"], ((data["3__kurtosis_x"]) * (data["flux_ratio_sq_sum"])) )) * (data["flux_ratio_sq_sum"]))))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["flux_err_mean"]>0, data["detected_flux_min"], data["flux_ratio_sq_sum"] ), data["2__kurtosis_y"] )) +
                0.100000*np.tanh((((((((data["flux_d0_pb4"]) < (np.where(data["flux_d0_pb2"] > -1, data["flux_d0_pb2"], data["flux_d0_pb4"] )))*1.)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"]<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], (-1.0*((np.where(data["hostgal_photoz_err"]<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )))) )) +
                0.100000*np.tanh(((data["4__skewness_x"]) + (np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["distmod"] )))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["1__kurtosis_y"], data["flux_err_mean"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["distmod"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])), data["1__fft_coefficient__coeff_1__attr__abs__x"] ) )))

    def GP_class_92(self,data):
        return (-1.730312 +
                0.100000*np.tanh(((data["flux_err_min"]) + (((((data["flux_err_min"]) + (-3.0))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(((((data["detected_mean"]) + (((data["detected_mean"]) + (-2.0))))) + (-2.0))) +
                0.100000*np.tanh(((((np.minimum(((((((data["flux_err_min"]) + (data["detected_mean"]))) * 2.0))), ((data["detected_mean"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((((((np.minimum(((np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["flux_err_min"]))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(np.where(((data["4__kurtosis_x"]) * 2.0) > -1, (-1.0*((data["4__kurtosis_x"]))), ((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh((((-1.0*((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), ((((np.minimum(((np.minimum(((((data["flux_err_min"]) * 2.0))), ((data["detected_mean"]))))), ((data["flux_err_min"])))) * 2.0))))) +
                0.100000*np.tanh((((((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((data["1__skewness_y"])))) + (-1.0))/2.0)) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (-2.0))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))))) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0))), ((((data["flux_err_min"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) + (((((((-1.0) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["4__kurtosis_y"]))))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.where(data["3__kurtosis_x"] > -1, -2.0, np.where(data["3__kurtosis_x"] > -1, data["detected_flux_diff"], data["detected_flux_std"] ) ), data["0__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__kurtosis_y"]))) * 2.0)) - (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((-1.0*((np.where(np.minimum(((data["3__skewness_x"])), ((data["3__kurtosis_x"]))) > -1, np.where(data["3__kurtosis_x"] > -1, 3.141593, data["3__kurtosis_x"] ), data["3__kurtosis_x"] ))))) +
                0.100000*np.tanh(np.where(-3.0<0, np.where(data["4__kurtosis_x"] > -1, ((-3.0) - (-2.0)), data["detected_mean"] ), data["flux_max"] )) +
                0.100000*np.tanh(((((np.minimum(((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_diff"])))), ((-3.0)))) + (np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["detected_flux_max"])))))) - (data["5__kurtosis_x"]))) +
                0.100000*np.tanh((-1.0*((np.where(np.where(data["flux_err_max"] > -1, data["4__kurtosis_x"], 2.718282 ) > -1, 2.0, data["5__kurtosis_x"] ))))) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_max"])), ((np.minimum(((data["detected_flux_err_min"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"]))))))) * 2.0)) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["3__kurtosis_x"]))))), ((((((data["flux_err_min"]) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(((((((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["detected_flux_err_min"])))) * 2.0)) * 2.0)) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["detected_flux_max"])), ((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))))))), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"]<0, np.where(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)>0, data["flux_err_min"], data["1__fft_coefficient__coeff_1__attr__abs__x"] ), ((data["flux_err_min"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, -3.0, np.where(data["3__kurtosis_y"] > -1, -3.0, ((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (-1.0)) ) )) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.minimum(((data["5__kurtosis_x"])), ((-3.0))), np.where(2.0 > -1, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, np.where(data["4__kurtosis_x"] > -1, -2.0, data["0__fft_coefficient__coeff_1__attr__abs__x"] ), data["detected_flux_std"] )) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]<0, data["flux_diff"], np.minimum(((-2.0)), ((np.minimum(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_diff"], data["3__skewness_x"] ))), ((data["distmod"])))))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((((data["flux_max"]) * 2.0)))))))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, np.where(data["hostgal_photoz"] > -1, -3.0, data["0__fft_coefficient__coeff_0__attr__abs__x"] ), data["0__fft_coefficient__coeff_0__attr__abs__y"] )) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, -3.0, (-1.0*((data["5__kurtosis_y"]))) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["flux_d1_pb2"]) * 2.0))))), ((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["2__fft_coefficient__coeff_1__attr__abs__y"])))))))), ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["4__kurtosis_x"])), ((data["detected_flux_min"])))))) - (data["1__kurtosis_y"]))) +
                0.100000*np.tanh(np.minimum(((((data["hostgal_photoz_err"]) + (np.minimum((((((data["flux_err_min"]) + (data["flux_err_min"]))/2.0))), ((np.minimum(((data["detected_flux_err_skew"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"])))))))))), ((data["detected_flux_max"])))) +
                0.100000*np.tanh(np.where(data["1__skewness_y"]>0, data["1__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["5__kurtosis_x"] > -1, -3.0, data["flux_err_min"] ) )) +
                0.100000*np.tanh(np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((((data["detected_mean"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((1.0))))))), ((data["flux_err_min"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.minimum(((data["detected_mean"])), ((data["detected_flux_max"]))))), ((data["flux_err_skew"])))))))), ((data["detected_flux_err_min"])))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["4__kurtosis_y"]))) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["flux_dif3"]))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_max"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["4__skewness_x"]<0, np.where(data["detected_flux_max"]<0, data["detected_flux_err_min"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ), -2.0 )) +
                0.100000*np.tanh(((((((-2.0) + (data["detected_flux_err_min"]))) + ((((data["2__skewness_x"]) + (data["1__skewness_y"]))/2.0)))) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((data["flux_err_min"])), ((((data["detected_flux_max"]) + (data["2__skewness_y"]))))))))) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["4__kurtosis_x"])))))) +
                0.100000*np.tanh((((((-1.0*((((np.where(data["4__kurtosis_x"] > -1, data["5__kurtosis_x"], 1.0 )) * (data["1__fft_coefficient__coeff_0__attr__abs__x"])))))) - (data["5__kurtosis_y"]))) - (data["flux_min"]))) +
                0.100000*np.tanh(((data["flux_dif3"]) + ((-1.0*((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0)) + ((((data["flux_err_median"]) > (data["2__fft_coefficient__coeff_1__attr__abs__y"]))*1.))))))))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(((((data["detected_mean"]) - (np.where(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )))) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -2.0, data["1__kurtosis_y"] )) +
                0.100000*np.tanh(np.where((-1.0*((((data["detected_flux_dif2"]) / 2.0)))) > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_max"]<0, (-1.0*(((-1.0*((data["2__fft_coefficient__coeff_0__attr__abs__x"])))))), np.where(data["4__skewness_x"]<0, (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__skewness_y"]))/2.0), -3.0 ) )) +
                0.100000*np.tanh((((((((np.minimum(((data["1__skewness_y"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) + (data["detected_flux_max"]))/2.0)) - (data["2__kurtosis_y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(np.minimum(((((data["flux_max"]) * (data["1__skewness_y"])))), ((data["detected_flux_std"])))>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], np.minimum(((data["detected_flux_max"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) )) +
                0.100000*np.tanh(((data["flux_err_skew"]) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, data["flux_err_skew"], -3.0 )) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_skew"]) * 2.0)) - (data["3__kurtosis_x"]))) +
                0.100000*np.tanh((-1.0*((np.where(((data["detected_flux_min"]) - (data["5__kurtosis_x"]))>0, np.where(data["detected_flux_min"]<0, data["detected_flux_skew"], data["detected_flux_min"] ), data["5__kurtosis_x"] ))))) +
                0.100000*np.tanh(((data["ddf"]) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, -2.0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_ratio_sq_sum"] ) )) +
                0.100000*np.tanh(((np.where(data["0__skewness_x"]>0, data["1__fft_coefficient__coeff_0__attr__abs__x"], ((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["4__fft_coefficient__coeff_0__attr__abs__y"])) )) - (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_ratio_sq_sum"] > -1, data["detected_flux_err_skew"], data["2__skewness_x"] ))), ((data["detected_flux_err_min"])))) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((np.where(data["5__kurtosis_x"] > -1, np.minimum(((3.0)), ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["mwebv"]))))), ((data["detected_mjd_size"]) / 2.0) ))))) +
                0.100000*np.tanh(np.minimum(((data["flux_dif3"])), ((np.minimum(((data["flux_dif3"])), ((data["detected_flux_err_std"]))))))) +
                0.100000*np.tanh(((((((data["1__skewness_y"]) + (np.where(data["1__skewness_y"] > -1, data["flux_err_std"], data["1__fft_coefficient__coeff_0__attr__abs__x"] )))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["1__skewness_y"]))) +
                0.100000*np.tanh((((((data["flux_d0_pb5"]) < (np.minimum(((data["flux_max"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))))*1.)) / 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_err_median"])), ((((((((data["1__kurtosis_y"]) < (data["2__fft_coefficient__coeff_1__attr__abs__x"]))*1.)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))/2.0))))) - (data["4__skewness_y"]))) +
                0.100000*np.tanh(((((data["2__skewness_y"]) * (((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (data["2__skewness_y"]))))) - ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["flux_d1_pb3"]) / 2.0)) * 2.0)))/2.0)))) +
                0.100000*np.tanh(np.where(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where((9.86017036437988281)>0, data["flux_d0_pb3"], data["flux_dif3"] ), (-1.0*((data["4__skewness_y"]))) ) > -1, data["detected_flux_err_skew"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.minimum(((((-1.0) / 2.0))), ((np.tanh((data["detected_mean"])))))) +
                0.100000*np.tanh((((((np.tanh((data["1__skewness_y"]))) - (data["flux_d1_pb3"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(((np.minimum(((np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["2__skewness_y"])))) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb5"] > -1, np.where(data["detected_flux_min"] > -1, data["detected_flux_min"], data["0__skewness_x"] ), data["detected_flux_err_skew"] )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_dif3"]) + (data["mwebv"])) )) +
                0.100000*np.tanh(((((np.where(data["4__skewness_x"]<0, data["1__skewness_y"], (((-1.0*((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0))))) / 2.0) )) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((data["detected_mean"]) - (np.where(data["flux_dif3"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], (((data["3__kurtosis_x"]) < (data["2__kurtosis_y"]))*1.) )))) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_dif3"]))), data["flux_d0_pb4"] )) - (data["detected_flux_mean"]))) +
                0.100000*np.tanh(np.where(np.maximum((((((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["detected_flux_by_flux_ratio_sq_sum"])))), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]))) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]>0, ((data["detected_flux_diff"]) + (np.where(data["flux_dif3"]>0, np.maximum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["detected_flux_err_skew"]))), -3.0 ))), data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, (((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["5__skewness_x"]))/2.0), data["mwebv"] )) +
                0.100000*np.tanh(((np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__kurtosis_x"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)))) +
                0.100000*np.tanh(np.where((-1.0*((np.tanh((data["detected_flux_std"])))))<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.where((((np.where(data["flux_ratio_sq_sum"] > -1, data["3__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_by_flux_ratio_sq_skew"] )) + (data["detected_flux_by_flux_ratio_sq_skew"]))/2.0)<0, data["flux_d1_pb4"], data["mwebv"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_mean"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["1__skewness_y"]<0, np.where(data["detected_mean"]>0, data["1__skewness_y"], data["1__skewness_y"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] ))), ((data["flux_d1_pb1"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"]>0, ((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__skewness_y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])), data["2__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((data["detected_flux_err_skew"]) - (np.where(((data["flux_diff"]) / 2.0) > -1, ((data["flux_diff"]) - (data["flux_d1_pb4"])), ((data["flux_diff"]) / 2.0) )))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (np.tanh((data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["flux_by_flux_ratio_sq_skew"], np.where(data["5__kurtosis_x"] > -1, -1.0, data["detected_flux_diff"] ) )) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) * (2.0))) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (np.maximum(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d0_pb3"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, np.minimum(((data["flux_err_skew"])), ((((data["detected_flux_err_max"]) - (data["2__kurtosis_y"]))))), data["flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(np.minimum(((data["0__kurtosis_y"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((-1.0*((np.where(data["flux_ratio_sq_skew"]>0, np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__y"])), (-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"]))) ))))) +
                0.100000*np.tanh(np.where((((data["flux_err_skew"]) > (((data["detected_flux_err_skew"]) / 2.0)))*1.)<0, data["1__fft_coefficient__coeff_0__attr__abs__x"], np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_err_skew"]))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((((data["flux_err_skew"]) * (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (-3.0)))))))) +
                0.100000*np.tanh((((((2.718282) / 2.0)) < (np.where(data["mwebv"]<0, data["flux_ratio_sq_sum"], (-1.0*(((-1.0*((data["mjd_size"])))))) )))*1.)) +
                0.100000*np.tanh((((data["flux_err_skew"]) + (data["detected_mean"]))/2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"] > -1, np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), (((((np.where(data["flux_max"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["flux_dif3"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))), -3.0 )) +
                0.100000*np.tanh(((np.minimum(((data["detected_mean"])), (((((((data["detected_flux_err_min"]) + (data["mwebv"]))/2.0)) / 2.0))))) / 2.0)) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, (-1.0*(((((data["5__skewness_x"]) < (data["5__kurtosis_x"]))*1.)))), (((data["flux_d1_pb5"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.) )) +
                0.100000*np.tanh((-1.0*((np.tanh(((-1.0*(((((((-1.0*((((np.tanh((data["1__skewness_y"]))) / 2.0))))) * (data["3__skewness_x"]))) * 2.0)))))))))) +
                0.100000*np.tanh(np.minimum(((data["1__skewness_x"])), ((data["0__kurtosis_y"])))) +
                0.100000*np.tanh(((((((data["flux_std"]) > (np.minimum(((data["5__skewness_y"])), ((data["flux_diff"])))))*1.)) < (np.minimum(((data["flux_err_min"])), ((data["1__skewness_x"])))))*1.)) +
                0.100000*np.tanh(((data["3__fft_coefficient__coeff_1__attr__abs__y"]) - ((-1.0*((data["4__kurtosis_y"])))))) +
                0.100000*np.tanh(((((((-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__y"])))) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) - (((data["1__kurtosis_x"]) * 2.0)))) +
                0.100000*np.tanh(np.where((2.0)>0, (((((data["flux_d1_pb3"]) < (data["3__kurtosis_y"]))*1.)) / 2.0), (((((data["flux_err_std"]) / 2.0)) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) )) +
                0.100000*np.tanh(((((np.tanh((np.tanh((((data["detected_flux_mean"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))))))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((np.minimum(((0.0)), ((data["flux_err_mean"])))) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_skew"]<0, data["1__skewness_y"], data["flux_d1_pb0"] )) - (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.tanh((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_median"]))))) +
                0.100000*np.tanh((-1.0*((np.where(data["flux_max"] > -1, ((((data["mjd_diff"]) - (np.tanh((data["flux_max"]))))) / 2.0), data["flux_dif3"] ))))) +
                0.100000*np.tanh(np.minimum((((((data["0__skewness_y"]) < (((np.tanh((data["flux_dif3"]))) * 2.0)))*1.))), ((np.minimum(((-1.0)), ((data["1__skewness_x"]))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]>0, 2.0, data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0), np.minimum(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))) )) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, np.where(data["4__kurtosis_x"]<0, data["detected_flux_by_flux_ratio_sq_skew"], ((-2.0) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])) ), ((-2.0) - (data["detected_flux_min"])) )) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__x"] )) / 2.0)) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["3__fft_coefficient__coeff_1__attr__abs__y"]))))), ((data["flux_d0_pb4"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_by_flux_ratio_sq_skew"] > -1, data["flux_max"], ((data["flux_max"]) - (((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__skewness_y"]))) / 2.0))) )) +
                0.100000*np.tanh(np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), (((((3.0)) * 2.0))))) +
                0.100000*np.tanh(((((3.141593) - (((((data["flux_ratio_sq_sum"]) * 2.0)) - (data["mwebv"]))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.where((((data["1__skewness_y"]) < (data["flux_err_skew"]))*1.) > -1, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["hostgal_photoz"] )))

    def GP_class_95(self,data):
        return (-1.890339 +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) * 2.0)) + (np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) * 2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["hostgal_photoz"]))) + (data["distmod"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((((((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)) + (data["flux_w_mean"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["distmod"]) * 2.0)))) * 2.0)) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["distmod"]) * 2.0)))) + (data["distmod"]))) + (((data["distmod"]) + (data["detected_flux_w_mean"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (((data["distmod"]) + (((data["hostgal_photoz"]) + (data["distmod"]))))))))) +
                0.100000*np.tanh(((((data["flux_d1_pb4"]) + (((data["detected_flux_min"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["distmod"]) + (((((np.minimum(((data["3__skewness_y"])), ((((data["flux_d1_pb0"]) * 2.0))))) / 2.0)) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) + (((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (np.minimum(((((data["flux_d1_pb5"]) + (data["hostgal_photoz_err"])))), ((data["distmod"])))))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["flux_d0_pb5"]) + (((data["hostgal_photoz"]) + (data["flux_d0_pb5"]))))))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (((data["flux_median"]) + (data["hostgal_photoz"]))))) * 2.0)) +
                0.100000*np.tanh((((((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))))/2.0)) + (data["flux_d0_pb3"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb5"])), ((np.minimum(((data["distmod"])), ((((data["flux_max"]) + (data["5__skewness_x"]))))))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["distmod"]) + ((((data["0__skewness_x"]) + (((data["distmod"]) * 2.0)))/2.0)))))) +
                0.100000*np.tanh(((((np.minimum(((((((data["hostgal_photoz"]) + (data["flux_d0_pb1"]))) * 2.0))), ((data["hostgal_photoz"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (((((np.minimum(((data["hostgal_photoz"])), ((data["5__skewness_x"])))) * 2.0)) + (data["flux_d0_pb5"]))))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((((data["flux_median"]) + (np.minimum(((np.minimum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))))), ((((((data["hostgal_photoz"]) * 2.0)) + (data["detected_flux_min"])))))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) + (((((data["distmod"]) * 2.0)) * 2.0))))), ((((data["flux_d0_pb5"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_d0_pb4"]) - (data["hostgal_photoz_err"]))) * 2.0)) + (((data["flux_d0_pb4"]) - (data["flux_d0_pb5"]))))) * 2.0)) +
                0.100000*np.tanh(((data["flux_d1_pb4"]) + (((data["detected_flux_min"]) + (data["4__skewness_x"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) * 2.0)) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (((data["hostgal_photoz_err"]) * 2.0)))) + (data["0__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb5"])), ((data["hostgal_photoz"])))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((((data["hostgal_photoz"]) * 2.0)))))), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))))), ((((data["hostgal_photoz"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((np.minimum(((data["1__skewness_x"])), ((((((data["flux_d1_pb5"]) * 2.0)) + (((data["1__skewness_x"]) * 2.0)))))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__skewness_x"]) + (data["detected_flux_min"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((((data["0__skewness_x"]) + (data["0__skewness_x"])))))) + (((data["0__skewness_x"]) + (((data["0__skewness_x"]) * 2.0)))))) +
                0.100000*np.tanh(((data["distmod"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["detected_flux_median"]) + (data["hostgal_photoz"]))) + (data["flux_std"]))) + (((np.minimum(((data["5__skewness_x"])), ((data["hostgal_photoz"])))) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((np.minimum(((data["4__skewness_x"])), ((data["hostgal_photoz"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) - (data["hostgal_photoz_err"])))), ((((data["hostgal_photoz"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["hostgal_photoz_err"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["hostgal_photoz"]))) + (data["flux_std"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"]))) * 2.0)) +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) + (data["hostgal_photoz"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["5__skewness_x"])), ((((((data["0__skewness_x"]) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["distmod"]) + (data["detected_flux_min"]))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((((data["detected_flux_err_mean"]) + (data["hostgal_photoz"]))) + (data["hostgal_photoz"])))), ((data["flux_d0_pb5"])))) + (((data["hostgal_photoz"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((((data["flux_d0_pb4"]) - (data["hostgal_photoz_err"]))))))), ((data["1__skewness_x"])))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz_err"]<0, data["distmod"], ((((data["hostgal_photoz"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"])) )) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_max"]))) + (((((data["hostgal_photoz"]) + (data["flux_std"]))) + (((data["hostgal_photoz"]) + (-1.0))))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(((data["2__kurtosis_x"]) / 2.0) > -1, data["hostgal_photoz_err"], (((((data["hostgal_photoz_err"]) * 2.0)) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) )))) +
                0.100000*np.tanh(((((data["flux_d1_pb5"]) + (np.minimum(((data["1__skewness_x"])), ((np.minimum(((data["flux_d0_pb4"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["flux_skew"])))))))))))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (((data["hostgal_photoz"]) + (((np.minimum(((data["flux_diff"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))) * 2.0)))))) + (((data["hostgal_photoz"]) * 2.0)))) +
                0.100000*np.tanh(((np.minimum(((((np.minimum(((data["detected_flux_min"])), ((((np.tanh((data["hostgal_photoz"]))) - (data["hostgal_photoz_err"])))))) * 2.0))), ((data["hostgal_photoz"])))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__kurtosis_x"]))) - (data["2__kurtosis_x"]))) - (data["2__kurtosis_x"]))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_skew"])), ((((((data["hostgal_photoz"]) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (data["hostgal_photoz"]))) - (((data["hostgal_photoz_err"]) * (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((np.where(data["3__skewness_x"]<0, data["flux_dif3"], ((data["flux_d0_pb0"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"])) )) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz_err"]) + (((data["flux_d1_pb0"]) + (np.minimum(((data["hostgal_photoz"])), ((data["detected_mjd_diff"])))))))))) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["flux_d1_pb4"], -3.0 )) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (((((((data["hostgal_photoz"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) + (data["hostgal_photoz"]))))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))) - (data["1__skewness_y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((((np.minimum(((data["flux_median"])), ((((data["flux_d0_pb5"]) - (data["flux_skew"])))))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))) * 2.0)) + (0.0))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["2__kurtosis_y"]) - (data["hostgal_photoz_err"]))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_skew"] > -1, data["flux_skew"], data["hostgal_photoz"] )) - (((data["hostgal_photoz_err"]) * (data["flux_skew"]))))) * (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["1__skewness_x"]) * 2.0))))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (data["detected_flux_err_mean"]))) + (data["3__skewness_x"]))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((((-1.0) + (((data["hostgal_photoz"]) * 2.0)))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))) )) * 2.0)) +
                0.100000*np.tanh(((((((((data["detected_flux_median"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) / 2.0)) + (data["0__skewness_x"]))) +
                0.100000*np.tanh((((((data["hostgal_photoz_err"]) < (data["4__fft_coefficient__coeff_1__attr__abs__y"]))*1.)) - ((((data["hostgal_photoz_err"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["hostgal_photoz_err"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))/2.0)))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (np.minimum(((((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))), ((data["flux_max"])))))) +
                0.100000*np.tanh(np.minimum(((np.tanh((np.minimum(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["hostgal_photoz_err"])))), ((data["flux_median"]))))))), ((data["flux_median"])))) +
                0.100000*np.tanh((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, data["1__kurtosis_x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )))/2.0)) - (data["hostgal_photoz_err"]))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"]))))), ((np.where(data["flux_median"]<0, data["detected_mjd_diff"], np.minimum(((data["detected_mjd_diff"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"]))) ))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["detected_flux_mean"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb0"])), ((np.minimum(((((np.minimum(((data["detected_mjd_size"])), ((data["3__fft_coefficient__coeff_0__attr__abs__y"])))) / 2.0))), ((data["1__skewness_x"]))))))) * 2.0)) +
                0.100000*np.tanh(((((data["2__kurtosis_y"]) + ((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["hostgal_photoz"]))/2.0)))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d0_pb3"]))))), ((((((data["flux_median"]) * 2.0)) * 2.0))))))))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["hostgal_photoz"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))) + (((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) + (data["hostgal_photoz_err"]))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + ((((data["0__skewness_y"]) + (((data["detected_flux_err_mean"]) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["hostgal_photoz"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))))))/2.0)))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"]>0, ((np.where(data["hostgal_photoz"] > -1, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_err_median"] )) - (data["hostgal_photoz_err"])), data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((((np.where(data["flux_skew"]<0, data["detected_flux_err_std"], data["hostgal_photoz"] )) + (data["4__skewness_y"]))) + (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (data["hostgal_photoz"]))) + (((((data["1__skewness_y"]) + (data["hostgal_photoz"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]>0, data["mjd_diff"], np.where(data["2__kurtosis_x"]>0, ((data["3__kurtosis_y"]) + (data["2__kurtosis_x"])), data["3__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]<0, data["1__skewness_x"], data["detected_flux_mean"] )) +
                0.100000*np.tanh(((data["flux_d1_pb3"]) + (((np.minimum(((data["2__kurtosis_y"])), ((data["detected_flux_max"])))) + (data["2__kurtosis_y"]))))) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, ((((data["flux_min"]) / 2.0)) - (data["1__skewness_y"])), data["1__skewness_x"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["flux_skew"]) + (np.minimum(((np.minimum(((data["flux_median"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"])))))))), ((data["flux_median"]))))))))) +
                0.100000*np.tanh((((data["flux_d0_pb5"]) + (data["flux_skew"]))/2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (data["detected_flux_err_median"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]<0, data["flux_d1_pb1"], ((np.minimum(((data["detected_mjd_diff"])), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) - (data["detected_mean"])) )) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]<0, (((((data["flux_d0_pb5"]) * 2.0)) + (data["1__skewness_x"]))/2.0), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, data["3__skewness_x"], np.where(data["hostgal_photoz"]<0, np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz_err"]))), ((data["3__skewness_x"]) - (data["hostgal_photoz_err"])) ) )) +
                0.100000*np.tanh((((data["flux_median"]) + (((data["detected_flux_err_median"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_dif3"]>0, np.where(np.where(data["flux_dif3"]>0, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_dif3"] )>0, data["flux_dif3"], data["flux_dif3"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((((((((((data["distmod"]) - (data["hostgal_photoz_err"]))) - (data["1__skewness_y"]))) - (data["1__skewness_y"]))) - (data["1__skewness_y"]))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((np.where(((data["3__kurtosis_y"]) * 2.0)<0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.tanh((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) + (data["hostgal_photoz"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_mjd_diff"])), ((data["2__kurtosis_y"]))))), ((data["2__kurtosis_y"])))) +
                0.100000*np.tanh(((data["detected_flux_err_std"]) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(np.where((-1.0*((data["hostgal_photoz_err"]))) > -1, data["hostgal_photoz"], (-1.0*((((data["detected_flux_err_min"]) + (data["hostgal_photoz"]))))) )) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((((data["hostgal_photoz_err"]) + (((data["hostgal_photoz"]) + (((data["hostgal_photoz_err"]) + (data["flux_max"]))))))) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.where(data["4__kurtosis_y"]>0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["flux_err_std"]>0, data["flux_err_min"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["3__kurtosis_y"] ), data["flux_median"] )) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((((data["flux_d1_pb0"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) + (((data["detected_mean"]) + (((data["hostgal_photoz_err"]) + (data["detected_flux_skew"]))))))) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]>0, np.tanh((np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - ((-1.0*((data["4__fft_coefficient__coeff_1__attr__abs__y"])))))>0, data["mjd_diff"], data["1__kurtosis_y"] ))), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((data["0__skewness_y"]) + (((data["3__skewness_x"]) + (np.minimum(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_median"])))), ((data["0__skewness_y"])))))))) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) + (((data["1__skewness_x"]) + (((((((data["1__kurtosis_x"]) + (data["1__skewness_x"]))) + (data["5__skewness_x"]))) - (data["hostgal_photoz_err"]))))))) +
                0.100000*np.tanh(((-1.0) + (((data["hostgal_photoz"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh(((((data["detected_flux_skew"]) + (data["4__skewness_y"]))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["flux_err_min"]))) + (((data["flux_err_mean"]) * (data["1__skewness_y"]))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(data["5__skewness_x"]>0, data["detected_flux_max"], data["flux_d1_pb5"] )))) / 2.0)) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_std"]<0, data["flux_d0_pb5"], data["flux_d0_pb5"] )) +
                0.100000*np.tanh(np.minimum(((((data["3__kurtosis_y"]) - (np.minimum(((data["0__skewness_y"])), ((3.0))))))), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, (-1.0*((data["4__kurtosis_x"]))), data["distmod"] )) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["hostgal_photoz_err"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz_err"]))))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_mjd_diff"]))))))))

































class GPLRV:
    def __init__(self):
        self.classes = 14
        self.class_names = [ 'class_6',
                             'class_15',
                             'class_16',
                             'class_42',
                             'class_52',
                             'class_53',
                             'class_62',
                             'class_64',
                             'class_65',
                             'class_67',
                             'class_88',
                             'class_90',
                             'class_92',
                             'class_95']


    def GrabPredictions(self, data):
        oof_preds = np.zeros((len(data), len(self.class_names)))
        oof_preds[:,0] = self.GP_class_6(data)
        oof_preds[:,1] = self.GP_class_15(data)
        oof_preds[:,2] = self.GP_class_16(data)
        oof_preds[:,3] = self.GP_class_42(data)
        oof_preds[:,4] = self.GP_class_52(data)
        oof_preds[:,5] = self.GP_class_53(data)
        oof_preds[:,6] = self.GP_class_62(data)
        oof_preds[:,7] = self.GP_class_64(data)
        oof_preds[:,8] = self.GP_class_65(data)
        oof_preds[:,9] = self.GP_class_67(data)
        oof_preds[:,10] = self.GP_class_88(data)
        oof_preds[:,11] = self.GP_class_90(data)
        oof_preds[:,12] = self.GP_class_92(data)
        oof_preds[:,13] = self.GP_class_95(data)
        oof_df = pd.DataFrame(np.exp(oof_preds), columns=self.class_names)
        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)
        return oof_df


    def GP_class_6(self,data):
        return (-1.965653 +
                0.100000*np.tanh(((((((((((data["flux_err_min"]) + ((((((data["0__kurtosis_y"]) < (data["flux_err_min"]))*1.)) * 2.0)))) / 2.0)) * 2.0)) * 2.0)) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (data["flux_err_min"]))) + (((data["flux_err_min"]) + (((((data["flux_err_min"]) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh(((data["flux_diff"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((((data["detected_flux_err_mean"]) * 2.0)) + (data["flux_err_min"]))) + (data["flux_err_median"]))) +
                0.100000*np.tanh(((data["flux_std"]) + (((data["4__skewness_x"]) + ((((((data["4__skewness_x"]) + (data["detected_flux_err_mean"]))) + (data["detected_flux_err_median"]))/2.0)))))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_mean"])), ((((((data["flux_err_median"]) * 2.0)) * 2.0)))))), ((data["flux_err_mean"])))) + (data["flux_err_mean"]))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((((data["flux_err_median"]) * 2.0))))) + (((data["flux_err_min"]) / 2.0)))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((((np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh(np.minimum(((np.where(data["4__kurtosis_x"] > -1, data["detected_flux_skew"], ((np.minimum(((data["detected_flux_skew"])), ((data["detected_flux_err_min"])))) * 2.0) ))), ((((data["detected_flux_err_min"]) + (data["detected_flux_skew"])))))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((((((data["flux_err_min"]) * 2.0)) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, data["5__kurtosis_x"], ((data["5__kurtosis_x"]) * 2.0) ))), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) * 2.0)) +
                0.100000*np.tanh((((data["detected_flux_err_min"]) + (((np.where((((5.20438098907470703)) + (data["detected_flux_err_median"]))>0, data["detected_flux_err_min"], data["detected_flux_err_min"] )) * 2.0)))/2.0)) +
                0.100000*np.tanh(((data["detected_flux_err_min"]) + (((np.minimum(((((((-1.0) - (data["detected_flux_min"]))) - (data["distmod"])))), ((data["detected_flux_min"])))) - (data["distmod"]))))) +
                0.100000*np.tanh(np.where(data["flux_err_min"]>0, data["5__kurtosis_x"], np.where(data["mwebv"] > -1, np.minimum(((((data["detected_flux_min"]) * 2.0))), ((((data["flux_err_min"]) * 2.0)))), data["flux_err_min"] ) )) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["detected_flux_min"])), ((data["flux_err_min"]))))), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_skew"])), ((((data["flux_err_min"]) * (((data["detected_flux_min"]) * (data["flux_err_min"])))))))) + (((data["flux_err_min"]) + (data["flux_err_min"]))))) +
                0.100000*np.tanh(((((data["4__skewness_x"]) + (np.minimum(((data["flux_err_min"])), ((np.minimum(((data["flux_err_min"])), ((data["4__skewness_x"]))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["4__skewness_x"])), ((((((((data["flux_err_min"]) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["flux_err_min"])), ((data["detected_flux_err_std"]))))))) +
                0.100000*np.tanh(((np.minimum(((data["5__kurtosis_x"])), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_err_min"]<0, data["flux_err_min"], ((data["flux_err_min"]) * 2.0) )) * 2.0)) + (np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["flux_err_min"]))) * 2.0)) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) * (data["detected_flux_max"]))) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz"] > -1, data["detected_flux_err_max"], data["5__kurtosis_x"] ) > -1, np.where(data["distmod"] > -1, data["distmod"], data["5__skewness_x"] ), data["5__kurtosis_x"] )) +
                0.100000*np.tanh(((data["flux_max"]) * (np.minimum(((np.minimum(((((data["detected_flux_min"]) + (data["detected_flux_skew"])))), ((data["flux_max"]))))), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((data["detected_flux_min"]) - (data["distmod"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -3.0, np.where(data["hostgal_photoz"] > -1, data["distmod"], (((-1.0*((-3.0)))) + (data["detected_flux_min"])) ) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((data["flux_err_min"])))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_skew"])), ((data["5__skewness_x"])))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((data["flux_err_min"]) * 2.0))), ((((((((np.minimum(((data["flux_err_min"])), ((data["detected_flux_min"])))) * 2.0)) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, np.where(np.where(data["distmod"] > -1, data["distmod"], 3.141593 ) > -1, -3.0, data["distmod"] ), data["5__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(-2.0 > -1, ((-2.0) - (data["distmod"])), ((-2.0) - (data["distmod"])) )) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (((((np.minimum(((data["flux_d0_pb4"])), ((data["detected_flux_max"])))) + (np.minimum(((data["flux_max"])), ((data["flux_err_min"])))))) * (data["flux_max"]))))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) * (((np.minimum(((((data["flux_d0_pb4"]) * (data["detected_flux_max"])))), ((data["detected_flux_skew"])))) * (data["detected_flux_max"]))))) - (data["detected_flux_max"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_flux_skew"])))))) * (((data["detected_flux_min"]) + (data["detected_flux_skew"]))))) +
                0.100000*np.tanh(np.where((((data["distmod"]) + (data["distmod"]))/2.0) > -1, -1.0, np.where(((data["distmod"]) / 2.0) > -1, -1.0, (-1.0*((data["distmod"]))) ) )) +
                0.100000*np.tanh(((((np.minimum(((((((-2.0) * 2.0)) - (data["distmod"])))), ((-2.0)))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["5__skewness_x"])), ((((data["flux_err_min"]) + ((((((-1.0) * 2.0)) + (data["detected_flux_skew"]))/2.0)))))))))) +
                0.100000*np.tanh(np.where(data["flux_d0_pb4"]<0, (((data["flux_by_flux_ratio_sq_skew"]) + (((data["detected_flux_max"]) * (data["flux_d0_pb4"]))))/2.0), data["detected_flux_max"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, np.where(-3.0 > -1, -3.0, -3.0 ), ((data["4__skewness_x"]) - (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(((((((((-3.0) - (data["distmod"]))) * 2.0)) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((np.where(((data["distmod"]) - (-2.0))<0, -2.0, ((((data["distmod"]) - (-2.0))) - (data["distmod"])) ))))) +
                0.100000*np.tanh((-1.0*((((data["distmod"]) + (((((((data["detected_mjd_diff"]) + (data["distmod"]))) + (3.141593))) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((((-1.0) - (data["distmod"]))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )) - (3.141593)), ((data["detected_flux_min"]) * (data["flux_max"])) )) +
                0.100000*np.tanh(np.where(3.141593 > -1, (-1.0*((np.where(((data["distmod"]) / 2.0) > -1, 2.0, data["distmod"] )))), ((-1.0) - (data["distmod"])) )) +
                0.100000*np.tanh(((((((data["flux_d0_pb4"]) * (data["flux_max"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.minimum(((((-2.0) - (data["distmod"])))), ((-1.0)))) - (data["distmod"]))) +
                0.100000*np.tanh(((data["detected_flux_max"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (((data["flux_max"]) * (((data["detected_flux_min"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))))))) +
                0.100000*np.tanh(((np.minimum((((((-1.0*(((((2.0) + (data["distmod"]))/2.0))))) * 2.0))), ((((data["detected_flux_err_std"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh((((((((data["4__skewness_y"]) * (((((data["0__skewness_x"]) * 2.0)) * 2.0)))) + (((data["0__skewness_x"]) - (data["detected_mjd_diff"]))))/2.0)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(((data["mwebv"]) - (data["flux_err_min"])) > -1, ((-3.0) * (((data["mwebv"]) - (data["flux_err_min"])))), data["flux_skew"] )) +
                0.100000*np.tanh(np.where((((data["detected_flux_err_std"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)>0, data["5__fft_coefficient__coeff_0__attr__abs__y"], np.tanh((data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) +
                0.100000*np.tanh(((((((((np.minimum(((-2.0)), ((-2.0)))) - (data["distmod"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_min"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["4__skewness_y"]))) + ((((3.0)) * (((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (data["detected_flux_min"]))))))))) +
                0.100000*np.tanh(((((((data["detected_flux_median"]) * (data["detected_flux_max"]))) + (data["4__skewness_y"]))) + (((((data["flux_min"]) * (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((data["5__skewness_y"]) - (data["detected_mjd_diff"])), ((data["detected_mjd_diff"]) - (data["detected_flux_max"])) )) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) + (np.minimum(((data["5__kurtosis_x"])), (((((data["4__skewness_x"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))/2.0))))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_err_skew"])), ((((((((data["detected_flux_err_skew"]) + (data["3__skewness_y"]))) + (data["detected_flux_err_skew"]))) + (((data["3__skewness_y"]) + (data["detected_flux_err_skew"])))))))) +
                0.100000*np.tanh((((((-1.0*((np.where(np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["flux_err_min"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ) > -1, data["distmod"], data["detected_flux_err_std"] ))))) - (data["detected_flux_err_std"]))) * 2.0)) +
                0.100000*np.tanh(((((((((np.where(data["detected_mjd_diff"] > -1, data["3__skewness_y"], data["detected_mjd_diff"] )) - (data["detected_mjd_diff"]))) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((data["flux_err_min"])), ((data["detected_flux_err_std"]))))), ((data["flux_err_min"])))) + (data["flux_err_min"]))) * 2.0)) +
                0.100000*np.tanh((((data["flux_d0_pb4"]) + (data["flux_d0_pb4"]))/2.0)) +
                0.100000*np.tanh(((3.0) - (np.where(((data["distmod"]) / 2.0) > -1, (14.57934379577636719), data["hostgal_photoz"] )))) +
                0.100000*np.tanh(((data["flux_min"]) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((np.where(data["mwebv"] > -1, ((data["flux_err_min"]) - (data["mwebv"])), data["2__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (np.where((6.0) > -1, np.where(data["hostgal_photoz"] > -1, (6.0), data["hostgal_photoz"] ), data["1__fft_coefficient__coeff_0__attr__abs__y"] )))) +
                0.100000*np.tanh(((np.minimum(((data["1__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, data["distmod"], np.maximum(((((data["detected_flux_err_std"]) / 2.0))), ((np.maximum(((data["detected_flux_err_std"])), ((data["distmod"])))))) )) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["5__skewness_y"], data["detected_mjd_diff"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_max"]>0, data["detected_flux_err_skew"], np.where(data["flux_max"]>0, np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, data["detected_flux_err_skew"], ((data["detected_flux_err_skew"]) * 2.0) ), data["detected_flux_err_max"] ) )) +
                0.100000*np.tanh((-1.0*((((2.718282) + (((data["distmod"]) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) - (data["detected_flux_max"]))) + (((data["flux_min"]) + (np.minimum(((data["1__skewness_x"])), ((data["1__skewness_x"])))))))) +
                0.100000*np.tanh((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((np.where(data["detected_flux_max"] > -1, data["detected_flux_min"], ((data["detected_flux_err_median"]) + (0.367879)) )) + (data["2__skewness_x"])), data["detected_flux_err_max"] )) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_err_max"] )) + (((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)) + (data["2__kurtosis_x"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) * (((np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["flux_d1_pb5"], np.where(data["2__skewness_y"]<0, data["flux_d0_pb0"], data["flux_std"] ) )) - (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((data["detected_flux_err_mean"]) * 2.0)) + (data["detected_flux_err_max"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) - (np.where(np.tanh((data["1__fft_coefficient__coeff_1__attr__abs__y"])) > -1, data["mwebv"], data["mwebv"] )))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((data["4__skewness_y"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["1__kurtosis_x"], data["flux_min"] )))/2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]>0, (((data["3__skewness_y"]) < (data["5__skewness_y"]))*1.), (((data["flux_median"]) + (((data["detected_flux_w_mean"]) + (((data["1__kurtosis_x"]) * 2.0)))))/2.0) )) +
                0.100000*np.tanh(np.maximum(((np.where(data["hostgal_photoz"] > -1, -2.0, np.maximum(((((data["flux_min"]) + (data["flux_err_min"])))), ((-2.0))) ))), ((-2.0)))) +
                0.100000*np.tanh(((((data["5__skewness_y"]) + (data["flux_err_min"]))) - (np.where(((data["detected_mjd_diff"]) + (((data["5__skewness_y"]) / 2.0))) > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d1_pb0"])), ((((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__kurtosis_x"], np.minimum(((data["detected_flux_err_skew"])), ((data["1__kurtosis_x"]))) )) * 2.0))))) +
                0.100000*np.tanh(((np.maximum(((((np.tanh((data["flux_d1_pb0"]))) * (data["mjd_diff"])))), ((data["detected_flux_err_skew"])))) * (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]<0, data["flux_d1_pb1"], np.tanh((((np.maximum(((data["detected_flux_err_max"])), ((((data["4__skewness_x"]) - (data["flux_err_skew"])))))) + (data["4__skewness_x"])))) )) +
                0.100000*np.tanh(((1.0) * (data["detected_flux_err_median"]))) +
                0.100000*np.tanh(((np.minimum(((((np.tanh((data["5__skewness_y"]))) * (data["2__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_by_flux_ratio_sq_sum"])))) * (np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_ratio_sq_skew"], data["2__skewness_y"] )))) +
                0.100000*np.tanh(((((((-2.0) - (data["distmod"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((-1.0*((((data["distmod"]) + (np.maximum(((2.0)), ((((data["distmod"]) + (data["distmod"]))))))))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]>0, ((data["1__skewness_x"]) * 2.0), np.where(((data["flux_dif2"]) * 2.0)<0, ((data["detected_flux_max"]) * (data["detected_flux_dif3"])), data["flux_err_skew"] ) )) +
                0.100000*np.tanh((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__kurtosis_x"]))/2.0)) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) * (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) * (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["5__skewness_y"], ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) ) )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((((data["3__kurtosis_y"]) + (data["flux_err_min"]))/2.0)) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((data["flux_d0_pb4"]) * (data["3__skewness_x"]))) +
                0.100000*np.tanh(np.where(((-2.0) - (data["distmod"]))<0, np.where(-2.0<0, ((-2.0) - (data["distmod"])), data["detected_flux_median"] ), 2.718282 )) +
                0.100000*np.tanh(((data["flux_median"]) + (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_min"]))) + (data["flux_min"]))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) > (data["0__kurtosis_x"]))*1.)) + (np.minimum(((data["flux_err_min"])), ((data["flux_err_min"])))))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"]<0, data["detected_flux_skew"], np.minimum(((data["flux_err_min"])), ((data["flux_err_min"]))) )) +
                0.100000*np.tanh(((data["0__kurtosis_x"]) + ((((((data["0__kurtosis_x"]) + (((data["3__skewness_y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["detected_mjd_diff"]) / 2.0)))) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_sum"]) * (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) * (data["detected_flux_by_flux_ratio_sq_sum"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) - (data["mwebv"]))) - (data["mwebv"]))) - (data["mwebv"]))) +
                0.100000*np.tanh(np.where(data["5__skewness_x"]<0, data["flux_err_skew"], ((data["1__kurtosis_x"]) * ((((data["detected_flux_ratio_sq_sum"]) > (np.maximum(((data["flux_mean"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))))*1.))) )) +
                0.100000*np.tanh(((np.maximum(((data["flux_median"])), ((data["flux_median"])))) * (((data["5__skewness_x"]) / 2.0)))) +
                0.100000*np.tanh(np.where(np.tanh((2.718282)) > -1, np.where(data["detected_flux_by_flux_ratio_sq_sum"]<0, np.where(data["flux_err_skew"] > -1, data["detected_flux_err_skew"], data["detected_flux_skew"] ), data["5__fft_coefficient__coeff_0__attr__abs__x"] ), data["mjd_diff"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_skew"]) * (np.tanh((data["detected_flux_err_skew"]))))) * (((data["4__skewness_x"]) * (data["detected_flux_err_skew"]))))) +
                0.100000*np.tanh(((((np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) * 2.0)) * (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["mwebv"]<0, data["flux_d1_pb2"], ((data["flux_err_min"]) - (data["mwebv"])) )) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]<0, ((((data["flux_err_skew"]) + (data["flux_err_skew"]))) + (data["detected_flux_median"])), ((data["detected_flux_median"]) + ((-1.0*((data["detected_flux_median"]))))) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["flux_d1_pb0"]>0, data["detected_flux_err_median"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["flux_err_min"], data["flux_err_min"] ) ), data["flux_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"] > -1, ((data["4__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0), np.tanh((data["detected_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"] > -1, data["flux_err_min"], data["flux_err_min"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))

    def GP_class_15(self,data):
        return (-1.349153 +
                0.100000*np.tanh(((data["flux_d1_pb0"]) + (np.minimum(((data["0__skewness_x"])), ((((((data["0__skewness_x"]) - (data["mjd_size"]))) + (data["distmod"])))))))) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) + (np.minimum(((((data["flux_d0_pb1"]) + (((data["flux_d1_pb0"]) + (data["flux_d1_pb0"])))))), ((data["5__kurtosis_y"])))))) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) + (data["distmod"]))) + (((((np.minimum(((data["0__skewness_x"])), ((data["flux_d0_pb0"])))) - (data["detected_mjd_size"]))) * 2.0)))) +
                0.100000*np.tanh(np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["distmod"]) + (((data["detected_flux_min"]) * 2.0))))))), ((data["flux_d0_pb0"])))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_d1_pb0"])), ((data["flux_d0_pb0"]))))), ((((np.minimum(((data["0__skewness_x"])), ((data["detected_flux_min"])))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["0__skewness_x"])), ((((data["flux_d1_pb0"]) * 2.0))))) * 2.0))), ((((data["flux_d1_pb0"]) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb1"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), (((-1.0*((data["5__fft_coefficient__coeff_0__attr__abs__y"])))))))))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["0__skewness_x"]) + (data["detected_flux_min"]))))) + (data["flux_err_std"]))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((np.minimum(((data["0__skewness_x"])), ((data["flux_d0_pb0"])))) + (((data["flux_d0_pb0"]) + (data["distmod"]))))))) +
                0.100000*np.tanh(np.minimum(((((data["flux_d0_pb0"]) + (data["5__kurtosis_y"])))), ((data["flux_d1_pb0"])))) +
                0.100000*np.tanh(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["1__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["1__skewness_x"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["flux_d1_pb1"]) - (1.0))) - (((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (((((data["flux_d1_pb0"]) - (data["ddf"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((((((data["2__kurtosis_x"]) + (np.tanh((data["0__skewness_x"]))))) + (data["detected_flux_min"]))) + (np.where(data["distmod"] > -1, data["ddf"], data["detected_flux_min"] )))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (((((data["distmod"]) + (((data["0__skewness_x"]) + (data["detected_flux_min"]))))) + (data["distmod"]))))) +
                0.100000*np.tanh((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["flux_d0_pb1"]))/2.0)) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((data["1__skewness_x"]) + (((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_flux_min"]) * 2.0)))) + (np.tanh((((data["distmod"]) + (data["flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((data["hostgal_photoz_err"])))) + (((((data["detected_flux_min"]) + (((data["distmod"]) + (data["mjd_diff"]))))) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((data["flux_ratio_sq_skew"]) + (np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"]))))))), ((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))) + (data["flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((np.minimum(((data["flux_ratio_sq_skew"])), ((data["distmod"])))) + (data["flux_ratio_sq_skew"]))) + (data["1__skewness_x"]))) + (np.minimum(((data["flux_err_median"])), ((data["flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) - (data["4__kurtosis_x"]))) + (np.minimum(((data["flux_d1_pb1"])), ((np.minimum(((data["flux_d0_pb0"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))))))))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) - (np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]<0, np.where(data["detected_mjd_size"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d0_pb0"] ), data["detected_mjd_size"] )))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["distmod"]))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + ((((((data["flux_ratio_sq_skew"]) + (np.minimum(((data["flux_err_std"])), ((data["distmod"])))))/2.0)) + (data["flux_err_median"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (np.minimum(((np.minimum(((((data["distmod"]) - (0.0)))), ((data["flux_d1_pb1"]))))), ((data["distmod"])))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["flux_ratio_sq_skew"]))))), ((data["1__skewness_x"]))))), ((data["mjd_diff"]))))), ((data["flux_ratio_sq_skew"])))) +
                0.100000*np.tanh(((data["distmod"]) + (((((data["mjd_diff"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["flux_d1_pb1"]) + (((((data["distmod"]) + (data["detected_flux_min"]))) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (np.minimum(((data["detected_flux_min"])), ((data["distmod"])))))))))) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["distmod"]))) + (data["1__skewness_x"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((data["detected_flux_by_flux_ratio_sq_skew"]) + (np.minimum(((((data["detected_flux_err_min"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) + (((data["distmod"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((((((data["4__skewness_x"]) - (data["ddf"]))) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_mean"]))))) - (data["ddf"]))) +
                0.100000*np.tanh(((data["flux_d0_pb1"]) - (np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["flux_d0_pb1"]))>0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["1__skewness_x"] )))) +
                0.100000*np.tanh(((((((data["detected_flux_ratio_sq_skew"]) - (data["ddf"]))) + (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_d1_pb1"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["ddf"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (data["detected_flux_ratio_sq_skew"]))) + (((data["distmod"]) + (data["4__kurtosis_x"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["flux_d0_pb1"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb1"]) + (np.minimum(((data["flux_err_mean"])), ((data["distmod"])))))) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) + (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_by_flux_ratio_sq_skew"])))) + (data["4__skewness_x"])))))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) - (data["flux_d0_pb0"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]>0, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"])), data["3__kurtosis_x"] ), data["3__skewness_x"] )) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((data["ddf"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["detected_flux_ratio_sq_skew"]) + (((data["distmod"]) + (((data["mjd_diff"]) + (((data["flux_d0_pb0"]) + (data["mjd_diff"]))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.tanh((data["4__kurtosis_x"]))) * (((((data["detected_flux_min"]) * (data["4__kurtosis_x"]))) + (((data["detected_flux_min"]) * (data["4__kurtosis_x"]))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["ddf"]))) + (((((data["flux_ratio_sq_skew"]) - (((data["ddf"]) - (data["detected_flux_min"]))))) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((np.minimum(((data["hostgal_photoz_err"])), ((data["distmod"])))) + (((data["hostgal_photoz_err"]) + (data["detected_flux_min"]))))))))) +
                0.100000*np.tanh(np.where(np.where(data["detected_flux_err_min"]<0, data["detected_flux_ratio_sq_skew"], data["5__kurtosis_x"] )<0, ((data["detected_flux_err_min"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["detected_flux_ratio_sq_sum"])), ((data["3__kurtosis_x"]))))), ((((data["distmod"]) + (data["flux_ratio_sq_skew"]))))))), ((data["detected_flux_min"])))) +
                0.100000*np.tanh(np.where((((data["flux_d0_pb0"]) < (data["flux_ratio_sq_sum"]))*1.)>0, data["detected_flux_err_min"], 0.367879 )) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * 2.0)) + (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((data["detected_mean"]) < (((data["flux_d1_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))*1.)) +
                0.100000*np.tanh(((((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_mean"]>0, np.where(data["hostgal_photoz_err"]>0, data["distmod"], np.where(data["detected_mean"]>0, data["distmod"], data["distmod"] ) ), data["flux_ratio_sq_sum"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_d0_pb1"], (((-1.0*((data["4__fft_coefficient__coeff_1__attr__abs__y"])))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["flux_ratio_sq_sum"]))) + (data["flux_d1_pb5"]))))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((-1.0*((data["3__fft_coefficient__coeff_0__attr__abs__y"])))) > (data["detected_flux_err_min"]))*1.)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["flux_d1_pb3"]) * (np.where(np.minimum(((data["ddf"])), ((data["5__skewness_x"]))) > -1, data["2__kurtosis_x"], np.where(data["5__skewness_x"] > -1, data["4__kurtosis_x"], data["5__skewness_x"] ) )))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, ((((data["flux_d0_pb0"]) - (data["detected_flux_std"]))) * 2.0), ((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_ratio_sq_sum"]) * 2.0))) )) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_diff"]>0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["4__kurtosis_x"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["4__kurtosis_x"] ) ), data["1__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((np.where(data["detected_flux_std"]>0, data["detected_mjd_diff"], ((data["2__kurtosis_x"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"])) )) + (np.where(data["2__kurtosis_x"]>0, data["2__kurtosis_x"], data["flux_d0_pb5"] )))) +
                0.100000*np.tanh(((np.where(np.minimum(((data["detected_flux_err_min"])), ((data["detected_flux_ratio_sq_sum"])))>0, data["4__kurtosis_x"], data["distmod"] )) * (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["mjd_diff"], np.maximum(((data["flux_dif2"])), ((((data["2__kurtosis_x"]) - (np.minimum(((data["flux_dif2"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))))) )) +
                0.100000*np.tanh((((data["distmod"]) + (np.where(data["detected_flux_err_min"]>0, data["5__skewness_x"], ((np.minimum(((data["5__skewness_x"])), ((data["3__kurtosis_x"])))) + (data["flux_d0_pb5"])) )))/2.0)) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_median"]))*1.)) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) + (data["flux_d1_pb2"]))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_flux_std"], ((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_flux_std"])) )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], data["flux_ratio_sq_sum"] ), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_min"]))) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh((((((data["hostgal_photoz_err"]) * 2.0)) + (np.where((((data["flux_ratio_sq_skew"]) + ((((data["hostgal_photoz_err"]) + (data["distmod"]))/2.0)))/2.0)>0, data["flux_ratio_sq_skew"], data["hostgal_photoz_err"] )))/2.0)) +
                0.100000*np.tanh(((np.where(np.where(data["4__kurtosis_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )<0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["detected_mjd_diff"] )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_d0_pb1"]) + (((data["flux_ratio_sq_skew"]) - (data["mwebv"]))))) - (data["ddf"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where((((data["4__kurtosis_x"]) + (data["3__kurtosis_x"]))/2.0)<0, ((data["4__kurtosis_x"]) - (data["detected_flux_skew"])), data["flux_d1_pb5"] )) * (data["detected_flux_skew"]))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (((np.minimum(((data["distmod"])), ((data["distmod"])))) * 2.0)))))) +
                0.100000*np.tanh(((((((np.where((-1.0*((data["detected_mjd_diff"]))) > -1, data["detected_mjd_diff"], (-1.0*((data["detected_mjd_diff"]))) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]>0, data["flux_d1_pb5"], ((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_max"]))) * 2.0)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.minimum((((((((data["flux_median"]) * 2.0)) + (data["flux_median"]))/2.0))), ((data["mjd_size"])))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, np.maximum(((((((data["flux_d0_pb1"]) * 2.0)) + (np.maximum(((data["flux_d0_pb1"])), ((data["2__kurtosis_x"]))))))), ((data["detected_flux_ratio_sq_sum"]))), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(((data["2__kurtosis_x"]) + (np.maximum(((((data["1__kurtosis_y"]) + (data["2__kurtosis_x"])))), ((((data["2__kurtosis_x"]) + (data["2__kurtosis_x"])))))))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["detected_flux_ratio_sq_sum"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) + (data["detected_flux_err_min"]))) + ((((data["detected_flux_max"]) < (data["detected_flux_min"]))*1.)))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_min"]<0, ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_dif2"]) * 2.0) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["flux_err_mean"]) - (np.where(data["flux_d1_pb1"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["4__kurtosis_y"] )))) + (data["5__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["detected_flux_ratio_sq_sum"]))) + (data["detected_mjd_diff"]))) + (((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((data["flux_ratio_sq_sum"])), ((data["3__kurtosis_x"])))))))) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb5"]<0, np.where(data["1__kurtosis_y"]<0, data["1__kurtosis_y"], data["distmod"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]>0, data["flux_ratio_sq_sum"], np.where(data["detected_flux_err_min"]>0, data["flux_ratio_sq_sum"], ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_ratio_sq_sum"]))) * 2.0) ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_median"] > -1, data["detected_mjd_diff"], data["detected_flux_std"] )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]>0, data["2__kurtosis_x"], (((data["flux_by_flux_ratio_sq_sum"]) > (data["4__fft_coefficient__coeff_0__attr__abs__y"]))*1.) )) +
                0.100000*np.tanh(((((((np.where(data["flux_err_median"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], np.maximum(((((data["flux_median"]) * 2.0))), ((data["detected_flux_err_min"]))) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["mjd_diff"], np.where(data["mjd_diff"]>0, data["distmod"], data["distmod"] ) )) +
                0.100000*np.tanh((((((data["flux_median"]) < (np.where(data["3__skewness_x"]<0, data["detected_mjd_diff"], data["detected_flux_ratio_sq_sum"] )))*1.)) * (np.where(data["1__kurtosis_y"]>0, data["flux_median"], data["1__kurtosis_y"] )))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * (data["detected_flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((np.maximum(((data["2__kurtosis_x"])), ((data["flux_d1_pb3"])))) + (data["3__skewness_x"]))) * (((data["2__kurtosis_x"]) * (data["flux_d1_pb2"]))))) +
                0.100000*np.tanh(((data["distmod"]) + ((((((data["distmod"]) + (data["distmod"]))) + (data["hostgal_photoz_err"]))/2.0)))) +
                0.100000*np.tanh(np.where(data["detected_flux_median"]>0, data["4__kurtosis_x"], np.where(data["0__kurtosis_x"]>0, data["detected_flux_ratio_sq_sum"], data["4__kurtosis_x"] ) )) +
                0.100000*np.tanh((((((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) > (np.where(data["flux_d1_pb0"]>0, data["detected_flux_std"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )))*1.)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) > (data["hostgal_photoz"]))*1.)) +
                0.100000*np.tanh(np.where(data["flux_diff"]>0, data["detected_flux_err_min"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["4__skewness_x"] )) +
                0.100000*np.tanh((((data["flux_d0_pb1"]) > (data["detected_flux_dif3"]))*1.)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, np.where(((data["flux_err_skew"]) * 2.0)<0, data["2__kurtosis_x"], data["flux_median"] ), data["flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) * (data["detected_flux_err_skew"]))) * 2.0)) +
                0.100000*np.tanh(((np.maximum(((data["detected_flux_median"])), ((data["1__skewness_y"])))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((data["flux_d0_pb2"]) > (np.where(np.maximum(((np.minimum(((data["flux_d0_pb2"])), ((data["detected_flux_dif3"]))))), ((data["flux_d0_pb2"]))) > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb2"] )))*1.)) +
                0.100000*np.tanh(np.where(data["4__skewness_x"] > -1, np.where(data["4__skewness_x"]<0, data["hostgal_photoz_err"], data["1__skewness_y"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((np.where(np.where(data["detected_mjd_diff"]>0, data["flux_d0_pb3"], data["flux_d1_pb0"] )>0, data["distmod"], data["flux_d0_pb1"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(((data["detected_flux_err_median"]) * 2.0)<0, data["flux_d1_pb0"], np.where(data["flux_d1_pb0"]>0, data["distmod"], data["flux_d1_pb5"] ) )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_median"]) * 2.0)) + (((data["flux_median"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((np.maximum(((data["detected_mjd_diff"])), ((data["flux_err_skew"])))) + (((data["flux_d1_pb0"]) + (((data["1__kurtosis_y"]) + (data["detected_mjd_diff"]))))))) + (data["flux_d1_pb0"]))))

    def GP_class_16(self,data):
        return (-1.007018 +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (((((data["flux_ratio_sq_sum"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["flux_skew"]) * 2.0)) + (data["flux_skew"])))))))))) +
                0.100000*np.tanh((((((-1.0*((((data["flux_skew"]) * 2.0))))) - (data["flux_skew"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((-1.0*((((((((((data["flux_skew"]) * 2.0)) + (data["4__skewness_x"]))) + ((1.0)))) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh((-1.0*((((1.0) + (((np.maximum(((data["flux_skew"])), ((data["2__skewness_x"])))) + (((data["flux_skew"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"])))))))))) +
                0.100000*np.tanh((((-1.0*((((data["2__skewness_x"]) * 2.0))))) - (((((data["4__skewness_x"]) + (data["4__skewness_x"]))) + (3.0))))) +
                0.100000*np.tanh((-1.0*((np.where(np.where(np.where(2.718282 > -1, data["3__skewness_x"], data["distmod"] ) > -1, 2.718282, data["distmod"] ) > -1, 3.0, data["2__skewness_x"] ))))) +
                0.100000*np.tanh((-1.0*((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) + (((data["detected_flux_mean"]) + (((data["flux_by_flux_ratio_sq_skew"]) + (((2.0) + (data["flux_ratio_sq_sum"])))))))))))) +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (data["detected_flux_by_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((-2.0) - (data["2__skewness_x"]))) - (((data["flux_by_flux_ratio_sq_skew"]) - (((((-2.0) + (data["2__skewness_x"]))) - (data["2__skewness_x"]))))))) +
                0.100000*np.tanh(((np.where(((data["flux_d0_pb2"]) + (((data["3__skewness_x"]) * 2.0))) > -1, -2.0, -2.0 )) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((-2.0) - (((((data["2__skewness_x"]) + (data["2__skewness_x"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))))))) - (2.718282))) +
                0.100000*np.tanh((-1.0*((((((data["flux_skew"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["3__skewness_x"])))))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_skew"]) - (data["flux_skew"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(-2.0 > -1, -2.0, np.where(((-2.0) - (data["2__skewness_x"]))<0, -2.0, ((-2.0) - (data["2__skewness_x"])) ) )) +
                0.100000*np.tanh(((((-3.0) - (((data["flux_skew"]) + (data["flux_skew"]))))) + (((data["flux_skew"]) - (data["flux_skew"]))))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"] > -1, -2.0, (((((((-2.0) > (-2.0))*1.)) - (data["2__skewness_x"]))) - (data["flux_skew"])) )) +
                0.100000*np.tanh(((((-2.0) - (np.where(-2.0 > -1, ((-2.0) - (data["2__skewness_x"])), data["2__skewness_x"] )))) * 2.0)) +
                0.100000*np.tanh((-1.0*((((data["flux_skew"]) + (np.where(data["flux_by_flux_ratio_sq_skew"] > -1, 2.718282, data["5__fft_coefficient__coeff_1__attr__abs__x"] ))))))) +
                0.100000*np.tanh((-1.0*((((((data["detected_flux_ratio_sq_sum"]) + (((data["2__skewness_x"]) / 2.0)))) + (data["4__skewness_x"])))))) +
                0.100000*np.tanh(((((-3.0) - (data["2__skewness_x"]))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_err_min"]) + (((data["flux_median"]) + (data["flux_err_min"]))))))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["detected_flux_err_std"]))) * 2.0)) +
                0.100000*np.tanh(((((-3.0) - (data["4__skewness_x"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"] > -1, ((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_err_min"]) - (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))) )) +
                0.100000*np.tanh(((np.where(np.minimum(((-2.0)), ((data["flux_median"])))<0, ((-2.0) - (data["1__skewness_x"])), (-1.0*((data["4__skewness_x"]))) )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((((data["detected_flux_err_min"]) - (((np.maximum(((data["3__skewness_x"])), ((data["flux_median"])))) + (data["1__skewness_x"])))))), ((-1.0)))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (data["flux_median"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) + (data["flux_err_min"]))) + (data["flux_err_min"]))) + (data["flux_err_min"]))) +
                0.100000*np.tanh((-1.0*((((data["2__skewness_x"]) - (np.minimum(((((-2.0) - (data["detected_flux_by_flux_ratio_sq_skew"])))), ((-2.0))))))))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_median"])), ((data["detected_mjd_diff"])))) - (data["flux_skew"]))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, -3.0, ((((data["4__kurtosis_x"]) - (data["2__skewness_x"]))) - (data["2__skewness_x"])) )) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["flux_err_min"]) + (((data["flux_err_min"]) - (data["flux_err_min"]))))) + (data["flux_d1_pb0"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((((((data["flux_err_min"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((-2.0)), ((((data["2__skewness_x"]) - (data["3__skewness_x"])))))) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_median"]) - (((data["4__skewness_x"]) - (((data["flux_skew"]) - (data["2__skewness_x"]))))))) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((-1.0)), ((data["flux_ratio_sq_skew"])))) - (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.minimum(((((data["detected_flux_err_min"]) + (data["flux_d0_pb0"])))), ((np.where(data["5__kurtosis_y"]<0, data["flux_d0_pb0"], data["detected_flux_err_min"] ))))) +
                0.100000*np.tanh(np.where(((data["flux_err_min"]) / 2.0)>0, data["detected_flux_err_min"], ((data["detected_mjd_diff"]) - (data["flux_err_min"])) )) +
                0.100000*np.tanh(((np.minimum(((data["flux_median"])), ((((data["flux_median"]) - (data["detected_flux_min"])))))) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, ((np.minimum(((data["3__skewness_x"])), ((((-3.0) - (data["3__skewness_x"])))))) * 2.0), data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((data["flux_err_min"]) + (np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["detected_flux_err_min"], (-1.0*((data["flux_min"]))) )))) +
                0.100000*np.tanh((((((((((data["flux_skew"]) - (data["flux_skew"]))) - (data["flux_skew"]))) > (data["1__skewness_x"]))*1.)) - (data["flux_skew"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_median"])))))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_err_min"]))))), ((data["flux_err_min"])))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((((np.minimum(((data["flux_err_min"])), ((data["detected_mjd_diff"])))) + (data["flux_err_min"]))) * 2.0)) + (np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_err_min"])))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["flux_median"], np.where(((data["flux_median"]) - (data["detected_flux_min"]))>0, ((data["flux_median"]) - (data["detected_flux_min"])), -3.0 ) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["detected_flux_diff"])), ((data["flux_err_min"]))))))) +
                0.100000*np.tanh(((((-1.0) - (np.where(data["1__skewness_x"]>0, (-1.0*((((-1.0) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))), data["detected_flux_skew"] )))) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((((np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["0__kurtosis_y"]) + (data["detected_mjd_diff"])), data["flux_ratio_sq_skew"] )) + (data["flux_err_min"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"] > -1, data["flux_d0_pb5"], (-1.0*((data["2__skewness_x"]))) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, ((data["flux_median"]) + (((np.minimum(((data["detected_flux_ratio_sq_skew"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) + (0.0)))), data["flux_d0_pb0"] )) +
                0.100000*np.tanh(((np.where(((data["2__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_flux_skew"])) > -1, -2.0, data["flux_ratio_sq_skew"] )) - (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((((-3.0) + (((-3.0) * (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__skewness_x"])))))))), ((2.718282)))) +
                0.100000*np.tanh(((((data["5__kurtosis_y"]) - ((((((data["1__skewness_x"]) - (data["flux_d0_pb5"]))) > (data["2__fft_coefficient__coeff_0__attr__abs__y"]))*1.)))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((data["flux_err_skew"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_ratio_sq_skew"])), ((data["0__kurtosis_x"])))) + (np.minimum(((data["detected_flux_std"])), ((data["flux_ratio_sq_sum"])))))) +
                0.100000*np.tanh(np.where(data["detected_flux_mean"]>0, -3.0, np.where(-3.0>0, data["detected_flux_mean"], data["detected_flux_std"] ) )) +
                0.100000*np.tanh(np.where((((np.where(-2.0 > -1, ((-3.0) - (data["4__fft_coefficient__coeff_0__attr__abs__x"])), data["1__fft_coefficient__coeff_0__attr__abs__y"] )) > (data["flux_err_min"]))*1.) > -1, data["flux_err_min"], data["mjd_diff"] )) +
                0.100000*np.tanh(((-2.0) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_d0_pb0"]) + (np.where(data["flux_err_min"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb3"] )))))/2.0)))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) + (((data["detected_mjd_diff"]) - (data["3__skewness_x"]))))) + (((data["flux_median"]) - (data["flux_median"]))))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(((data["flux_median"]) + ((((((data["flux_median"]) + (data["detected_mjd_diff"]))/2.0)) - (data["2__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))) - (((data["flux_dif2"]) - (data["2__skewness_x"]))))) * 2.0)) +
                0.100000*np.tanh(((((-3.0) + (((np.tanh(((-1.0*((data["detected_flux_by_flux_ratio_sq_sum"])))))) * 2.0)))) / 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_min"] > -1, ((data["flux_d0_pb5"]) - (data["flux_mean"])), data["flux_mean"] )) - (((((data["detected_flux_min"]) * 2.0)) - (data["flux_mean"]))))) +
                0.100000*np.tanh(((np.where(((data["flux_err_min"]) / 2.0)>0, data["flux_err_min"], ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0) )) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((np.minimum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_mjd_diff"])))) + ((((-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["4__kurtosis_y"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_flux_err_min"])), ((data["flux_err_min"]))))), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((np.where(data["distmod"]>0, data["flux_median"], ((((data["flux_median"]) - (data["detected_flux_err_max"]))) * 2.0) )) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.tanh((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((data["flux_mean"]) > (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0)))*1.)) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]>0, -3.0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, data["1__fft_coefficient__coeff_1__attr__abs__x"], -3.0 ) )) +
                0.100000*np.tanh((((((-1.0*(((((data["flux_d1_pb5"]) < (data["detected_flux_median"]))*1.))))) - (data["detected_flux_w_mean"]))) - (data["detected_flux_median"]))) +
                0.100000*np.tanh(((np.where((-1.0*((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_ratio_sq_sum"])))))>0, data["flux_ratio_sq_skew"], data["0__kurtosis_x"] )) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.tanh((data["flux_ratio_sq_skew"]))) + ((((data["detected_mjd_diff"]) + (data["2__kurtosis_y"]))/2.0)))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["detected_mjd_diff"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((((((((data["flux_max"]) + (data["flux_ratio_sq_skew"]))/2.0)) * 2.0)) + (data["detected_mjd_diff"]))/2.0)) + (data["flux_ratio_sq_skew"]))/2.0)) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["detected_flux_std"]) + (data["detected_flux_std"]))))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.where(data["flux_ratio_sq_skew"]<0, data["detected_mjd_diff"], data["flux_err_min"] ))), ((data["flux_max"]))))))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), (((((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_max"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))/2.0))))) +
                0.100000*np.tanh(np.where(((data["flux_dif3"]) - (data["1__skewness_x"]))>0, (-1.0*((data["detected_flux_median"]))), data["flux_d0_pb0"] )) +
                0.100000*np.tanh(((np.where(data["flux_ratio_sq_skew"] > -1, ((data["detected_mjd_diff"]) - (data["detected_flux_err_std"])), data["flux_err_min"] )) + (data["flux_err_min"]))) +
                0.100000*np.tanh((((data["flux_ratio_sq_skew"]) + ((((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))/2.0)))/2.0)) +
                0.100000*np.tanh(((data["mjd_size"]) + (np.where(((data["detected_flux_err_max"]) + (((data["flux_ratio_sq_skew"]) + (data["detected_flux_err_min"])))) > -1, data["detected_flux_err_min"], data["flux_skew"] )))) +
                0.100000*np.tanh((((data["4__kurtosis_x"]) + ((((((data["5__skewness_y"]) + (data["flux_err_min"]))) + (data["detected_flux_max"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.minimum((((((((data["flux_by_flux_ratio_sq_skew"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["2__fft_coefficient__coeff_0__attr__abs__x"])))), ((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["detected_flux_err_min"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]<0, data["flux_err_min"], ((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__skewness_y"])))) - (data["1__skewness_x"])) )) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__x"])), ((np.where(data["flux_diff"]>0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["detected_mjd_diff"] ))))) +
                0.100000*np.tanh((((((-1.0*((((data["flux_d1_pb5"]) - (data["detected_flux_err_mean"])))))) - ((-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((-1.0*((((((np.maximum(((data["flux_d0_pb1"])), ((data["flux_d0_pb0"])))) - (data["ddf"]))) - (data["1__kurtosis_x"])))))) / 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_min"] > -1, data["0__kurtosis_y"], ((data["2__kurtosis_y"]) + (0.0)) )) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_err_min"])), ((((np.minimum(((((data["detected_flux_diff"]) - ((((((data["flux_err_min"]) + (data["flux_err_min"]))/2.0)) / 2.0))))), ((data["detected_flux_err_min"])))) * 2.0))))) +
                0.100000*np.tanh(np.where(data["detected_flux_dif2"] > -1, np.where(data["detected_flux_dif2"]>0, data["2__kurtosis_y"], data["flux_d1_pb0"] ), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((((np.where(data["detected_mjd_diff"] > -1, -3.0, ((data["2__skewness_y"]) * 2.0) )) + (data["3__skewness_y"]))) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (np.where(data["detected_flux_diff"] > -1, np.where(np.minimum(((data["3__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_mjd_diff"]))) > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_ratio_sq_skew"] ), data["flux_median"] )))) +
                0.100000*np.tanh((((((data["mjd_diff"]) - (np.minimum(((data["detected_flux_err_skew"])), ((data["detected_flux_err_mean"])))))) + (data["flux_ratio_sq_skew"]))/2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh((-1.0*((np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, np.maximum(((((data["2__skewness_x"]) - (data["1__kurtosis_x"])))), ((data["2__skewness_x"]))), data["ddf"] ))))) +
                0.100000*np.tanh((((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["flux_err_min"]) + (data["detected_mjd_diff"]))))/2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["flux_err_min"])), ((data["flux_ratio_sq_skew"])))) - (-2.0))) + (((data["flux_err_min"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["1__fft_coefficient__coeff_0__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["flux_dif3"] > -1, ((data["flux_dif3"]) - (data["flux_err_std"])), ((data["4__kurtosis_y"]) * 2.0) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["flux_dif3"])), ((data["0__kurtosis_x"]))))), ((data["flux_dif3"]))))), ((data["flux_dif3"])))) +
                0.100000*np.tanh(((np.where(data["flux_err_min"]>0, data["5__kurtosis_x"], np.minimum(((data["detected_flux_err_min"])), ((data["flux_dif3"]))) )) + (((data["flux_dif3"]) + (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(((((data["detected_flux_err_min"]) + (((data["flux_d0_pb5"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) + ((((data["detected_flux_max"]) + (data["detected_flux_err_min"]))/2.0)))) +
                0.100000*np.tanh((((((((data["detected_flux_err_mean"]) + (data["flux_d1_pb5"]))/2.0)) - (data["2__skewness_x"]))) + ((-1.0*((data["3__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where((-1.0*((data["2__fft_coefficient__coeff_0__attr__abs__y"]))) > -1, data["4__skewness_y"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(np.where((-1.0*((data["detected_flux_std"])))>0, data["4__skewness_y"], data["flux_std"] )>0, data["detected_flux_std"], data["4__skewness_y"] )) / 2.0)) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) - (np.minimum(((data["hostgal_photoz_err"])), ((data["4__fft_coefficient__coeff_1__attr__abs__x"])))))) + (np.tanh((((data["0__kurtosis_x"]) / 2.0)))))/2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_std"]>0, data["flux_err_skew"], ((data["flux_skew"]) * 2.0) )) +
                0.100000*np.tanh(((data["4__skewness_y"]) + ((((data["4__skewness_y"]) + (data["detected_flux_std"]))/2.0)))) +
                0.100000*np.tanh((((data["0__kurtosis_x"]) + (data["detected_mjd_diff"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), (((((np.maximum(((data["flux_dif3"])), ((data["ddf"])))) + (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d0_pb0"]))))/2.0)))))), ((data["2__fft_coefficient__coeff_0__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"] > -1, ((((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d1_pb2"])))) - (data["flux_err_std"]))) * (data["flux_max"])), data["3__skewness_x"] )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) * (data["detected_flux_err_min"]))))

    def GP_class_42(self,data):
        return (-0.859449 +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((-2.0))))), ((-3.0))))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((-2.0)), ((-2.0)))) +
                0.100000*np.tanh(np.minimum(((-3.0)), ((np.minimum(((np.minimum(((data["flux_mean"])), ((np.minimum(((data["detected_flux_min"])), ((-2.0)))))))), ((-3.0))))))) +
                0.100000*np.tanh(((np.minimum(((((np.minimum(((data["distmod"])), ((data["detected_flux_min"])))) + (data["flux_d1_pb4"])))), ((data["flux_d1_pb3"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["distmod"]) * 2.0))), ((data["detected_flux_min"]))))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_x"])), ((((np.minimum(((data["flux_min"])), ((data["flux_min"])))) * 2.0))))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) / 2.0))), ((np.tanh((-3.0)))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((data["2__skewness_x"]))))), ((data["flux_d0_pb5"])))) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((data["flux_min"])), ((np.minimum(((data["detected_flux_by_flux_ratio_sq_sum"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))))))) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_min"]>0, data["detected_flux_min"], data["flux_min"] ))), ((data["detected_flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_d1_pb2"])), ((np.minimum(((-2.0)), ((data["flux_dif2"]))))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (np.tanh((data["3__skewness_x"]))))) + (np.minimum(((data["detected_flux_min"])), ((((((data["distmod"]) + (data["flux_d0_pb4"]))) * 2.0))))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["distmod"])), ((np.minimum(((data["flux_dif2"])), ((np.where(np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["flux_by_flux_ratio_sq_skew"])))<0, data["flux_diff"], data["distmod"] )))))))) +
                0.100000*np.tanh(((((((((data["detected_flux_min"]) - (data["flux_ratio_sq_skew"]))) - (data["hostgal_photoz_err"]))) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["flux_d0_pb4"])))) + (((data["detected_flux_min"]) + (data["detected_flux_min"]))))) +
                0.100000*np.tanh(((((data["flux_min"]) + (np.minimum(((data["detected_flux_min"])), ((data["detected_flux_min"])))))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) - (np.minimum(((data["distmod"])), ((data["distmod"]))))))), ((np.minimum(((data["mjd_size"])), ((np.minimum(((data["distmod"])), ((data["distmod"])))))))))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["distmod"])))) + (((data["distmod"]) + (data["1__kurtosis_x"]))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (data["flux_d0_pb5"]))) + (np.minimum(((data["1__kurtosis_x"])), ((data["detected_flux_min"])))))) +
                0.100000*np.tanh(((np.where(np.maximum(((data["detected_flux_std"])), (((((data["detected_flux_min"]) > (data["flux_d0_pb4"]))*1.)))) > -1, data["detected_flux_min"], data["detected_flux_ratio_sq_skew"] )) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["5__kurtosis_y"]) + (data["5__kurtosis_y"])))), ((data["detected_flux_min"]))))), ((data["flux_min"])))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["distmod"]))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d1_pb2"]) / 2.0))), ((((data["detected_flux_min"]) + (((data["hostgal_photoz_err"]) + (data["detected_flux_min"])))))))) + (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((((((data["detected_flux_std"]) + (data["flux_diff"]))) / 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) / 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (((((data["detected_flux_min"]) + (((data["flux_min"]) + (data["hostgal_photoz_err"]))))) * 2.0)))) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) - (data["flux_diff"]))) - (data["flux_diff"]))) +
                0.100000*np.tanh(((data["0__kurtosis_x"]) - (np.maximum(((data["5__skewness_x"])), ((data["flux_diff"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["flux_mean"]) + (((data["detected_flux_dif3"]) - (data["flux_max"]))))) - (data["flux_max"]))) +
                0.100000*np.tanh((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)) +
                0.100000*np.tanh((((-1.0*((((data["detected_flux_std"]) - (((data["0__kurtosis_x"]) - (np.where(data["flux_mean"]>0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["detected_flux_std"] ))))))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((data["0__kurtosis_y"]) + (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0), ((((data["flux_d0_pb4"]) + (data["detected_mean"]))) + (data["flux_d0_pb4"])) )) +
                0.100000*np.tanh((((((data["4__skewness_x"]) + (((((data["detected_flux_max"]) + (((data["detected_flux_min"]) * 2.0)))) + (data["detected_flux_min"]))))/2.0)) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((data["1__kurtosis_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((((1.0) > (data["detected_flux_std"]))*1.)) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) + (np.maximum(((data["5__skewness_x"])), ((data["flux_d0_pb5"])))))) +
                0.100000*np.tanh(((((((((((data["detected_flux_ratio_sq_skew"]) - (data["detected_mean"]))) - (data["3__kurtosis_x"]))) - (data["detected_flux_err_std"]))) - (data["3__kurtosis_x"]))) - (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(((data["flux_d0_pb4"]) + (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) < (data["1__skewness_y"]))*1.)) - (data["detected_flux_skew"]))) +
                0.100000*np.tanh(np.tanh((((data["4__kurtosis_y"]) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))))) +
                0.100000*np.tanh(np.tanh((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (np.maximum(((((data["flux_mean"]) - (data["flux_dif3"])))), ((data["5__skewness_x"])))))))))) +
                0.100000*np.tanh((-1.0*((((data["3__kurtosis_x"]) + (data["distmod"])))))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (data["1__kurtosis_y"]))) +
                0.100000*np.tanh(((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh((((data["0__kurtosis_x"]) + ((((((data["detected_flux_min"]) - (((data["detected_flux_min"]) + (data["3__kurtosis_y"]))))) + (data["distmod"]))/2.0)))/2.0)) +
                0.100000*np.tanh(((((((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__fft_coefficient__coeff_0__attr__abs__x"])))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["detected_flux_skew"]))) +
                0.100000*np.tanh(((np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_dif2"], (((data["detected_mjd_diff"]) > (data["detected_flux_dif2"]))*1.) )) * 2.0)) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_ratio_sq_skew"]))) + (((np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["5__kurtosis_x"]) + (data["flux_by_flux_ratio_sq_skew"])))))) + (data["5__skewness_x"]))))) +
                0.100000*np.tanh(((((((((data["flux_d1_pb0"]) - (data["detected_flux_std"]))) + (data["hostgal_photoz"]))) - (np.tanh((data["flux_d1_pb0"]))))) * 2.0)) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["2__skewness_x"]) * (data["detected_flux_min"]))) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]>0, data["4__kurtosis_y"], data["flux_d1_pb0"] )) +
                0.100000*np.tanh((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_diff"]<0, np.maximum(((np.where(data["flux_diff"]<0, data["detected_flux_ratio_sq_skew"], data["5__kurtosis_y"] ))), ((data["detected_flux_ratio_sq_skew"]))), ((data["5__kurtosis_y"]) + (data["hostgal_photoz_err"])) )) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]>0, data["1__kurtosis_x"], ((np.where(data["flux_ratio_sq_skew"] > -1, data["1__kurtosis_x"], (-1.0*((data["1__kurtosis_x"]))) )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh((((np.where(data["flux_d0_pb4"]<0, data["detected_flux_dif2"], data["detected_mjd_diff"] )) > (data["detected_flux_std"]))*1.)) +
                0.100000*np.tanh(((data["detected_mjd_size"]) + (((data["detected_flux_min"]) + (data["flux_diff"]))))) +
                0.100000*np.tanh(((data["detected_flux_min"]) + (data["mjd_diff"]))) +
                0.100000*np.tanh((((data["flux_max"]) < (np.maximum((((((((data["detected_flux_std"]) + (data["0__kurtosis_x"]))) < (data["3__fft_coefficient__coeff_0__attr__abs__y"]))*1.))), ((data["3__kurtosis_x"])))))*1.)) +
                0.100000*np.tanh(((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_d1_pb0"]) + (data["distmod"]))))) +
                0.100000*np.tanh(np.where(((data["detected_mjd_diff"]) - (data["detected_flux_w_mean"]))>0, np.where((((data["hostgal_photoz"]) > (data["hostgal_photoz"]))*1.)<0, data["flux_max"], data["detected_flux_w_mean"] ), data["hostgal_photoz"] )) +
                0.100000*np.tanh(((data["flux_d1_pb1"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["3__kurtosis_y"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_d1_pb5"]))) + (data["2__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(((data["detected_flux_std"]) - (data["detected_flux_std"]))>0, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) - (data["detected_flux_std"])) )) +
                0.100000*np.tanh(np.where(np.where(data["2__skewness_x"]<0, data["2__skewness_x"], np.maximum(((data["2__skewness_x"])), ((data["flux_skew"]))) )<0, -3.0, (13.35520839691162109) )) +
                0.100000*np.tanh(np.where(data["flux_d0_pb0"]<0, np.where(data["flux_d0_pb0"]<0, data["flux_d0_pb0"], data["flux_d0_pb0"] ), ((data["flux_d0_pb1"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((data["flux_mean"]) + (((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_skew"])))) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.maximum(((data["1__kurtosis_x"])), (((((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) > (2.0))*1.)) - (data["4__fft_coefficient__coeff_0__attr__abs__y"])))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + ((((data["flux_d1_pb0"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)))/2.0)))) +
                0.100000*np.tanh(((((((data["0__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) * (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (np.maximum(((((np.maximum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_d0_pb5"])))) + (data["detected_flux_by_flux_ratio_sq_sum"])))), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(np.tanh((data["distmod"]))>0, data["flux_ratio_sq_skew"], np.where(data["hostgal_photoz"]>0, np.where(data["3__kurtosis_y"]>0, data["detected_flux_min"], data["3__fft_coefficient__coeff_0__attr__abs__x"] ), data["detected_flux_diff"] ) )) +
                0.100000*np.tanh(np.where(((data["distmod"]) * 2.0) > -1, np.where(data["hostgal_photoz"]<0, data["hostgal_photoz"], ((data["3__kurtosis_y"]) * 2.0) ), ((np.tanh((data["flux_d0_pb0"]))) * 2.0) )) +
                0.100000*np.tanh((((np.where(data["3__skewness_x"]>0, data["detected_mjd_diff"], data["flux_d0_pb3"] )) > (data["detected_flux_std"]))*1.)) +
                0.100000*np.tanh((((((((((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__kurtosis_y"]))) * 2.0)) < (((data["flux_d1_pb3"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)))))*1.)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["3__skewness_x"]) + (((data["detected_mjd_diff"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]<0, data["4__fft_coefficient__coeff_0__attr__abs__y"], ((((np.where(((data["distmod"]) * 2.0) > -1, data["hostgal_photoz"], data["4__kurtosis_x"] )) * 2.0)) * 2.0) )) * 2.0)) +
                0.100000*np.tanh((((data["distmod"]) + ((((((((((data["detected_flux_min"]) + (data["distmod"]))/2.0)) + (data["distmod"]))/2.0)) + (data["detected_flux_min"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_err_skew"]))) + (data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["2__skewness_y"])) )) +
                0.100000*np.tanh(np.where((((data["flux_by_flux_ratio_sq_sum"]) + (data["2__kurtosis_y"]))/2.0)>0, data["detected_flux_min"], np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]>0, data["3__kurtosis_y"], data["3__kurtosis_y"] ) )) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) * (data["0__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["0__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)))) + ((((data["distmod"]) + (((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__skewness_x"]))))/2.0)))) +
                0.100000*np.tanh(np.where(data["flux_dif2"]>0, np.where(data["hostgal_photoz"]>0, data["detected_flux_ratio_sq_skew"], data["distmod"] ), data["mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__skewness_x"], ((data["detected_flux_dif3"]) + (data["1__skewness_y"])) )) +
                0.100000*np.tanh(np.where(data["2__skewness_y"]>0, data["flux_d0_pb4"], np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb0"], data["2__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"]>0, data["detected_flux_min"], data["distmod"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_min"]) + (data["1__kurtosis_x"]))) * (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_sum"]) + (data["flux_d1_pb3"]))) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_w_mean"]))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["detected_flux_min"], data["detected_flux_min"] )) +
                0.100000*np.tanh(np.where(data["distmod"]>0, data["5__kurtosis_x"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.maximum((((((data["detected_mjd_diff"]) + (data["detected_flux_err_max"]))/2.0))), ((data["mjd_diff"])))) +
                0.100000*np.tanh(np.minimum(((((((data["flux_err_skew"]) - (np.tanh((data["mwebv"]))))) / 2.0))), ((data["flux_err_skew"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_dif2"] > -1, ((((((data["detected_mjd_diff"]) > (np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif2"] )))*1.)) > (data["detected_flux_dif2"]))*1.), data["detected_flux_dif2"] )) +
                0.100000*np.tanh(((np.where(data["detected_mean"]>0, data["flux_dif2"], data["distmod"] )) + (np.maximum(((data["distmod"])), ((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])))))))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["3__kurtosis_y"], np.where(data["flux_d0_pb0"]>0, np.where(data["flux_d0_pb1"]>0, data["3__kurtosis_y"], data["flux_d0_pb0"] ), data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh((((data["detected_flux_w_mean"]) + (((data["detected_flux_err_std"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))/2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"]>0, data["5__skewness_y"], (((data["5__skewness_y"]) > (((((((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) + (data["5__skewness_y"]))/2.0)))*1.) )) +
                0.100000*np.tanh((((((((((data["flux_d1_pb1"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["distmod"]))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"]<0, (((((data["1__skewness_x"]) > (data["1__skewness_x"]))*1.)) / 2.0), data["flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_skew"]<0, data["0__skewness_x"], np.where(data["hostgal_photoz"] > -1, ((data["hostgal_photoz"]) * 2.0), data["hostgal_photoz"] ) )) +
                0.100000*np.tanh((((((((data["flux_d1_pb3"]) - (data["0__kurtosis_y"]))) - (data["0__kurtosis_y"]))) + (((data["flux_d1_pb3"]) * 2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["hostgal_photoz_err"], np.where(data["flux_dif2"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["flux_dif2"] > -1, data["detected_flux_err_median"], data["detected_flux_err_median"] ) ) )) +
                0.100000*np.tanh(np.maximum(((((data["1__kurtosis_x"]) * (data["1__kurtosis_x"])))), ((data["1__kurtosis_x"])))) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_err_std"])), ((data["flux_d1_pb1"])))) +
                0.100000*np.tanh(np.tanh((((((((np.minimum(((((data["detected_mjd_diff"]) / 2.0))), ((data["3__skewness_y"])))) + (data["0__kurtosis_y"]))/2.0)) + (data["0__kurtosis_y"]))/2.0)))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["flux_d0_pb0"], np.where(data["flux_dif2"]>0, data["flux_dif2"], data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]<0, data["5__kurtosis_x"], data["4__fft_coefficient__coeff_0__attr__abs__x"] ), data["mjd_diff"] )) +
                0.100000*np.tanh((((((((data["5__kurtosis_y"]) * ((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh((((data["distmod"]) > (data["detected_mjd_diff"]))*1.)) +
                0.100000*np.tanh(np.where(data["2__skewness_y"] > -1, np.where(((data["detected_mjd_diff"]) + (data["2__kurtosis_y"])) > -1, np.maximum(((data["detected_mjd_diff"])), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))), data["detected_flux_dif3"] ), data["1__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], ((((data["4__skewness_x"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d0_pb3"], data["2__kurtosis_x"] ))) )) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) * (((((((data["3__skewness_y"]) + (data["2__kurtosis_y"]))) + (data["3__skewness_y"]))) + (((data["2__kurtosis_y"]) + (data["5__skewness_x"]))))))) +
                0.100000*np.tanh(((np.where(data["flux_dif3"]<0, data["detected_mjd_diff"], ((data["0__kurtosis_y"]) - (data["detected_mjd_diff"])) )) - (data["0__kurtosis_y"]))) +
                0.100000*np.tanh((-1.0*((np.where(((data["hostgal_photoz_err"]) * 2.0) > -1, (-1.0*((np.where(data["detected_flux_w_mean"]>0, data["flux_dif2"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )))), data["flux_diff"] ))))) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_sum"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"] > -1, data["1__skewness_x"], (((data["1__fft_coefficient__coeff_1__attr__abs__x"]) > (data["flux_d1_pb0"]))*1.) )))

    def GP_class_52(self,data):
        return (-1.867467 +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_sum"]) + (((((2.718282) + (data["flux_by_flux_ratio_sq_skew"]))) + (np.tanh((data["flux_by_flux_ratio_sq_sum"]))))))) +
                0.100000*np.tanh(((data["flux_min"]) + (((((data["3__skewness_x"]) + (data["flux_min"]))) + (data["flux_min"]))))) +
                0.100000*np.tanh((((((((((data["flux_min"]) + (((data["flux_min"]) + (data["flux_min"]))))) + (data["flux_d1_pb3"]))) + (data["flux_min"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(((((data["flux_min"]) + (((data["flux_min"]) + (data["3__fft_coefficient__coeff_0__attr__abs__x"]))))) + (((data["2__skewness_x"]) + (data["2__skewness_x"]))))) +
                0.100000*np.tanh(np.where(data["flux_min"] > -1, data["flux_min"], ((((data["flux_min"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_by_flux_ratio_sq_skew"])) )) +
                0.100000*np.tanh(((((data["flux_min"]) + (((((data["4__kurtosis_x"]) - (data["flux_diff"]))) * 2.0)))) + (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.where(np.tanh((data["2__fft_coefficient__coeff_1__attr__abs__y"]))>0, data["flux_ratio_sq_skew"], data["2__skewness_x"] )) + (((data["mjd_size"]) - (data["detected_flux_err_mean"]))))) +
                0.100000*np.tanh(np.minimum(((data["3__kurtosis_x"])), ((np.minimum(((data["2__kurtosis_x"])), ((data["distmod"]))))))) +
                0.100000*np.tanh(((((np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["5__kurtosis_x"])), ((data["5__kurtosis_x"]))))))) * 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((data["flux_min"]) + (((data["5__kurtosis_x"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d0_pb3"]) + (((np.where(data["4__kurtosis_x"]<0, data["flux_ratio_sq_skew"], data["flux_diff"] )) / 2.0))))), ((data["hostgal_photoz_err"])))) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((((data["flux_d0_pb2"]) / 2.0))), ((np.minimum(((data["flux_min"])), ((data["flux_err_skew"]))))))) +
                0.100000*np.tanh(((0.367879) * (data["0__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["flux_ratio_sq_skew"]))) - (((data["distmod"]) * (data["flux_err_std"]))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb3"])), ((np.minimum(((data["2__skewness_x"])), (((((data["2__kurtosis_x"]) < (data["1__kurtosis_x"]))*1.)))))))) +
                0.100000*np.tanh(np.minimum(((((data["4__kurtosis_x"]) * 2.0))), ((data["distmod"])))) +
                0.100000*np.tanh((((((((-1.0*((data["flux_diff"])))) - (data["flux_diff"]))) - (((data["flux_diff"]) - (data["5__kurtosis_x"]))))) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((data["4__kurtosis_x"])), ((data["flux_d0_pb3"]))))))) +
                0.100000*np.tanh(np.minimum(((data["0__kurtosis_x"])), ((data["5__kurtosis_x"])))) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) - (data["detected_flux_err_median"]))) + (((data["flux_std"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_std"]))))))) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb3"])), ((((((data["4__skewness_x"]) - (((data["detected_flux_dif3"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (data["hostgal_photoz_err"]))) + (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["1__skewness_x"]) + (np.where(data["mjd_diff"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["flux_d0_pb2"] )))) + (((data["flux_ratio_sq_skew"]) + (data["3__skewness_y"]))))) +
                0.100000*np.tanh((((((np.where(data["detected_flux_w_mean"]<0, data["flux_median"], data["flux_max"] )) < (data["flux_max"]))*1.)) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["distmod"]))) + (1.0))) +
                0.100000*np.tanh(((((data["detected_flux_diff"]) - (data["flux_err_max"]))) - (((data["detected_flux_std"]) * 2.0)))) +
                0.100000*np.tanh((((data["flux_diff"]) < (((data["flux_d0_pb2"]) - (data["flux_max"]))))*1.)) +
                0.100000*np.tanh(((((((((-1.0*((np.tanh((data["flux_d0_pb4"])))))) > (data["detected_flux_std"]))*1.)) * 2.0)) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((0.367879) + (((np.minimum(((((np.tanh((((data["flux_d0_pb2"]) + (data["detected_flux_min"]))))) + (data["flux_d0_pb3"])))), ((data["detected_flux_min"])))) * 2.0)))) +
                0.100000*np.tanh(((((((data["flux_median"]) + (data["flux_median"]))) + (data["flux_median"]))) + (((data["5__kurtosis_y"]) * (data["flux_median"]))))) +
                0.100000*np.tanh(((((data["1__kurtosis_y"]) + (((data["flux_median"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_d0_pb0"]) / 2.0)) / 2.0)) - (data["detected_flux_diff"]))) - (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(((((((data["flux_median"]) + (((data["2__kurtosis_x"]) + (data["flux_ratio_sq_skew"]))))) + (np.tanh((data["flux_median"]))))) + (2.718282))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["flux_d1_pb2"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["3__skewness_y"]) - (data["flux_d0_pb0"])), data["distmod"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]<0, data["flux_d0_pb4"], ((data["2__kurtosis_x"]) * 2.0) )) +
                0.100000*np.tanh(((((((data["flux_median"]) * 2.0)) + (((((((data["flux_median"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((((((((data["hostgal_photoz_err"]) - (data["flux_d0_pb0"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_d0_pb0"]))) - (((data["flux_d0_pb0"]) - (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((np.maximum(((data["distmod"])), ((data["flux_median"])))) + (((1.0) + (data["distmod"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["0__kurtosis_x"])), ((((data["flux_median"]) * 2.0)))))), ((data["4__kurtosis_x"])))) +
                0.100000*np.tanh(((np.tanh((((data["flux_mean"]) - (data["5__fft_coefficient__coeff_0__attr__abs__x"]))))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_median"]) - (data["detected_flux_std"]))) + (data["flux_by_flux_ratio_sq_skew"]))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((((np.where(data["flux_d0_pb2"]>0, data["flux_d0_pb2"], data["flux_d0_pb2"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (data["detected_flux_diff"]))) +
                0.100000*np.tanh(((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["5__kurtosis_x"] )) - (data["detected_flux_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d0_pb2"]>0, data["5__kurtosis_x"], data["5__kurtosis_x"] )) +
                0.100000*np.tanh(((data["flux_median"]) + ((((np.minimum(((data["5__kurtosis_y"])), ((np.tanh((data["flux_err_max"])))))) + (((data["flux_err_max"]) + (data["1__kurtosis_y"]))))/2.0)))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]>0, data["2__kurtosis_x"], np.where(data["flux_d1_pb1"]>0, data["ddf"], np.where(data["flux_d1_pb1"]>0, data["flux_d1_pb1"], (-1.0*((data["flux_d1_pb1"]))) ) ) )) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) - (((data["detected_flux_std"]) - (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__skewness_x"], data["flux_d1_pb4"] )))))) - (data["flux_d0_pb0"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]>0, ((data["flux_median"]) * 2.0), data["mwebv"] )) +
                0.100000*np.tanh(((((((((((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__kurtosis_x"])))) + (data["2__kurtosis_x"]))) * (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(np.where(data["flux_median"]<0, data["detected_flux_err_mean"], data["detected_flux_err_mean"] ) > -1, ((((data["mwebv"]) * 2.0)) * 2.0), data["3__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb2"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]<0, data["flux_d0_pb0"], (((data["flux_d0_pb0"]) < (data["5__kurtosis_y"]))*1.) )) +
                0.100000*np.tanh(((((((np.where(data["flux_d0_pb0"]>0, data["4__kurtosis_x"], ((data["flux_median"]) - (data["2__kurtosis_x"])) )) - (data["flux_d0_pb0"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((((data["flux_d0_pb2"]) > (data["flux_d1_pb1"]))*1.)) * 2.0)) + (((data["flux_d0_pb2"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(data["flux_dif3"]<0, np.minimum(((((data["flux_ratio_sq_skew"]) + (data["flux_d1_pb2"])))), ((((data["flux_ratio_sq_skew"]) * 2.0)))), data["mwebv"] )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb2"]>0, data["0__kurtosis_x"], data["4__skewness_y"] )) + (np.maximum(((data["4__skewness_y"])), ((data["flux_err_max"])))))) +
                0.100000*np.tanh(((((((data["mjd_diff"]) * (np.where(data["flux_d0_pb1"]<0, data["3__kurtosis_x"], np.where(data["flux_d0_pb1"]<0, data["mjd_diff"], data["5__fft_coefficient__coeff_0__attr__abs__x"] ) )))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["2__kurtosis_x"]) * (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.where((((data["detected_flux_max"]) < (data["flux_d0_pb2"]))*1.)>0, (((data["flux_d1_pb1"]) < (data["flux_d0_pb2"]))*1.), data["flux_err_std"] )) +
                0.100000*np.tanh(np.minimum(((((((data["flux_median"]) * (data["flux_by_flux_ratio_sq_skew"]))) * ((9.0))))), ((data["flux_d0_pb4"])))) +
                0.100000*np.tanh(np.where(data["flux_err_max"]>0, data["mjd_diff"], np.where(data["mjd_diff"]>0, data["0__kurtosis_x"], data["2__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(((((np.where(data["flux_ratio_sq_skew"]<0, data["flux_err_max"], ((data["flux_d0_pb2"]) + (data["distmod"])) )) * 2.0)) + (data["flux_d0_pb4"]))) +
                0.100000*np.tanh(((((np.where(data["0__skewness_x"]<0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["detected_mean"]>0, data["2__kurtosis_x"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) ) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((((((data["detected_flux_diff"]) > (data["flux_dif2"]))*1.)) - (((data["detected_flux_diff"]) - (data["flux_ratio_sq_skew"]))))) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * (((data["detected_flux_err_median"]) + (((((data["detected_flux_err_median"]) + (data["2__skewness_y"]))) + (data["2__skewness_y"]))))))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb1"]<0, ((data["flux_d0_pb0"]) * 2.0), data["2__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, data["detected_flux_min"], np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["4__fft_coefficient__coeff_1__attr__abs__y"] ) )) +
                0.100000*np.tanh(((((((((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_d0_pb0"]))) * 2.0)) - (data["2__skewness_y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)>0, data["4__kurtosis_x"], data["flux_d0_pb4"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["0__skewness_x"]>0, np.where(data["0__skewness_x"]>0, ((data["flux_d1_pb0"]) + (data["flux_err_max"])), ((data["flux_err_max"]) + (data["detected_flux_err_median"])) ), data["3__skewness_y"] )) +
                0.100000*np.tanh(((((data["2__kurtosis_x"]) * (((((((data["5__kurtosis_x"]) * (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * (((data["2__skewness_x"]) / 2.0)))))) * 2.0)) +
                0.100000*np.tanh((((((data["flux_median"]) > (data["2__skewness_y"]))*1.)) + (np.where(data["distmod"]>0, data["0__kurtosis_y"], ((data["distmod"]) + (data["flux_median"])) )))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((np.minimum(((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["4__kurtosis_x"]))))), ((data["2__kurtosis_x"]))))), ((data["flux_d1_pb1"])))) * 2.0)) * (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_std"]<0, data["flux_d0_pb3"], np.where(data["flux_median"]<0, data["0__kurtosis_x"], data["1__kurtosis_y"] ) )) +
                0.100000*np.tanh(((((((np.where(data["detected_flux_std"]>0, np.where(data["detected_flux_min"]<0, (-1.0*((data["3__fft_coefficient__coeff_0__attr__abs__y"]))), data["detected_flux_skew"] ), data["3__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["4__skewness_y"]) - (((data["4__kurtosis_x"]) + (data["4__kurtosis_x"])))), data["4__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]>0, data["4__kurtosis_x"], np.where(data["distmod"]>0, np.where(data["hostgal_photoz_err"]<0, data["detected_mean"], data["hostgal_photoz_err"] ), data["0__skewness_x"] ) )) +
                0.100000*np.tanh(np.where(data["flux_std"]>0, np.where(data["hostgal_photoz_err"]>0, data["distmod"], (-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__x"]))) ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(np.where(np.where(data["flux_err_median"]<0, data["detected_mean"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )<0, data["flux_d1_pb0"], data["2__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]>0, np.where(np.tanh((data["4__kurtosis_y"]))>0, data["2__fft_coefficient__coeff_0__attr__abs__y"], data["1__kurtosis_y"] ), ((data["flux_d1_pb4"]) + (data["detected_mean"])) )) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, np.where(data["1__kurtosis_y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], ((data["1__skewness_x"]) + (data["flux_median"])) ), data["1__kurtosis_y"] )) +
                0.100000*np.tanh(((data["2__kurtosis_y"]) * (np.where(np.tanh((data["2__skewness_y"])) > -1, data["flux_diff"], data["2__skewness_y"] )))) +
                0.100000*np.tanh(((data["flux_d1_pb4"]) + (np.where(data["detected_flux_err_mean"] > -1, np.where(data["0__kurtosis_x"] > -1, data["4__kurtosis_y"], data["flux_d1_pb4"] ), data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) - (data["3__skewness_x"]))) * (((data["2__skewness_y"]) + (np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((data["2__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0), data["3__skewness_x"] )))))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["2__kurtosis_x"]) * (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["distmod"]))) * 2.0)))) * 2.0)))))) +
                0.100000*np.tanh(((data["flux_err_max"]) + (((((np.where(data["detected_mean"]>0, data["2__kurtosis_x"], ((data["flux_err_median"]) + (data["2__kurtosis_x"])) )) * 2.0)) + (data["detected_mean"]))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_mean"]<0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__x"], data["flux_ratio_sq_skew"] ), data["flux_ratio_sq_skew"] ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, ((data["detected_flux_skew"]) * (np.where(data["detected_flux_min"]>0, data["detected_flux_ratio_sq_skew"], data["detected_flux_err_median"] ))), ((data["detected_flux_err_median"]) * (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"]<0, data["flux_median"], np.where(data["flux_median"]<0, data["1__skewness_x"], data["5__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["3__skewness_y"], np.where(data["3__skewness_y"]<0, data["0__skewness_x"], np.where(data["0__kurtosis_x"]>0, data["0__kurtosis_x"], data["2__kurtosis_x"] ) ) )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, np.where(data["detected_flux_err_mean"] > -1, np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), (((4.0)))), (4.0) ), (4.0) )) +
                0.100000*np.tanh(((((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["distmod"], data["detected_flux_err_max"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_err_skew"] > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["detected_flux_err_skew"] ), ((data["detected_flux_err_skew"]) * 2.0) )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"]<0, ((data["3__kurtosis_x"]) * (((((data["flux_d0_pb0"]) * 2.0)) * 2.0))), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb0"]>0, data["3__kurtosis_x"], ((((data["4__skewness_y"]) - (data["3__kurtosis_x"]))) - (data["3__kurtosis_x"])) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["mjd_size"]>0, ((np.where(data["mjd_size"]>0, data["1__kurtosis_y"], ((data["detected_flux_err_min"]) + (data["flux_err_mean"])) )) + (data["flux_err_mean"])), data["3__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(data["0__skewness_x"]>0, ((np.where(data["0__skewness_x"]>0, data["hostgal_photoz_err"], data["flux_d1_pb2"] )) - (data["flux_d1_pb2"])), data["flux_d1_pb2"] )) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0), np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb2"], data["5__kurtosis_x"] ) )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]<0, ((((data["3__kurtosis_x"]) - (data["4__kurtosis_x"]))) - (data["2__skewness_y"])), data["2__skewness_y"] )) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_mean"] > -1, (((((data["flux_err_max"]) > (data["2__fft_coefficient__coeff_0__attr__abs__x"]))*1.)) - (data["mjd_size"])), data["2__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh((-1.0*((((np.where(data["0__skewness_x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["2__skewness_y"] ), data["4__kurtosis_y"] )) * (((data["2__skewness_y"]) * 2.0))))))) +
                0.100000*np.tanh(((((data["2__kurtosis_x"]) * (((((data["flux_d1_pb2"]) + (data["distmod"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(data["1__kurtosis_x"]<0, np.where(data["detected_flux_max"]<0, data["detected_flux_max"], data["detected_flux_max"] ), ((((((data["detected_flux_max"]) < (data["detected_mjd_size"]))*1.)) < (data["flux_d0_pb1"]))*1.) )) +
                0.100000*np.tanh(((np.where(data["flux_err_max"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["distmod"]<0, data["detected_flux_err_skew"], (((-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0) ) )) * 2.0)) +
                0.100000*np.tanh((((11.19984531402587891)) * (((data["distmod"]) + ((((data["flux_w_mean"]) < ((((data["5__kurtosis_x"]) > ((((data["flux_w_mean"]) > (data["5__kurtosis_x"]))*1.)))*1.)))*1.)))))) +
                0.100000*np.tanh(((np.where(data["0__kurtosis_x"]<0, data["flux_err_max"], np.where(data["0__kurtosis_x"]<0, data["detected_flux_err_std"], np.where(data["flux_err_max"]<0, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_err_std"] ) ) )) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["detected_flux_err_median"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * (data["hostgal_photoz"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((((np.where(data["flux_ratio_sq_skew"]<0, data["3__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]<0, data["distmod"], data["2__fft_coefficient__coeff_0__attr__abs__y"] ) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mean"]>0, np.where(data["detected_mean"]<0, data["2__skewness_y"], data["2__skewness_y"] ), np.where(data["flux_ratio_sq_sum"]>0, (-1.0*((data["2__skewness_y"]))), data["detected_mean"] ) )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["hostgal_photoz_err"], np.maximum(((data["flux_d1_pb0"])), ((np.where(data["flux_ratio_sq_skew"]<0, data["hostgal_photoz_err"], np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_d1_pb0"]))) )))) )) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d1_pb0"], data["5__fft_coefficient__coeff_0__attr__abs__y"] ) ) )) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["1__skewness_y"], ((((((data["flux_ratio_sq_skew"]) - (data["1__skewness_y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["flux_dif2"])) )) +
                0.100000*np.tanh(((((np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, ((data["flux_ratio_sq_skew"]) * 2.0), ((data["4__skewness_y"]) - (data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_d1_pb2"])))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]>0, data["hostgal_photoz_err"], np.where(data["flux_d1_pb4"] > -1, data["flux_d1_pb4"], data["flux_d1_pb4"] ) )) * 2.0), 3.141593 )) +
                0.100000*np.tanh(((np.where(data["flux_err_skew"]>0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_err_skew"], data["1__fft_coefficient__coeff_0__attr__abs__x"] ), np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__kurtosis_x"] ) )) * 2.0)))

    def GP_class_53(self,data):
        return (-2.781493 +
                0.100000*np.tanh(((data["flux_err_std"]) + (np.minimum(((data["2__skewness_y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_mean"], data["detected_flux_err_median"] )) +
                0.100000*np.tanh((((((((((((data["flux_err_mean"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))/2.0)) + (data["5__skewness_y"]))/2.0)) + (-1.0))/2.0)) + (data["flux_err_std"]))) +
                0.100000*np.tanh(np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["3__skewness_y"])))) +
                0.100000*np.tanh((((data["flux_err_std"]) + (np.maximum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((((data["detected_mean"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))))))))/2.0)) +
                0.100000*np.tanh(((np.minimum(((data["5__skewness_y"])), ((((data["flux_err_mean"]) + (data["detected_mean"])))))) + (np.where(data["detected_flux_diff"] > -1, data["flux_err_mean"], -2.0 )))) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), ((((np.where(data["2__skewness_y"] > -1, data["3__skewness_y"], data["flux_err_max"] )) + (((data["1__kurtosis_x"]) / 2.0))))))) +
                0.100000*np.tanh(((((0.0) + (data["5__skewness_y"]))) + (np.where(-1.0<0, data["flux_err_mean"], data["flux_d1_pb1"] )))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + ((((np.minimum(((data["detected_flux_std"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) + ((((data["5__skewness_y"]) + (data["flux_max"]))/2.0)))/2.0)))) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), (((((data["detected_mean"]) + (data["flux_err_std"]))/2.0))))) +
                0.100000*np.tanh(((((data["flux_err_std"]) + (data["flux_err_std"]))) + (data["3__skewness_y"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (-3.0))) + (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"] > -1, np.where(data["flux_err_std"] > -1, data["flux_err_std"], data["5__fft_coefficient__coeff_1__attr__abs__y"] ), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((np.where(data["flux_err_std"]>0, data["flux_err_std"], data["3__skewness_y"] )) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["detected_mean"] > -1, data["flux_err_std"], data["3__skewness_y"] )) +
                0.100000*np.tanh(((((data["5__skewness_y"]) + (((data["5__skewness_y"]) * 2.0)))) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) + (((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((((((data["flux_err_std"]) + (data["flux_err_std"]))) + (((data["3__skewness_y"]) + (((data["1__kurtosis_x"]) / 2.0)))))) + (-2.0))) +
                0.100000*np.tanh(np.minimum(((np.minimum((((((data["3__skewness_y"]) + (data["flux_err_std"]))/2.0))), ((data["detected_mean"]))))), ((((data["4__skewness_y"]) + (data["4__skewness_y"])))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((-3.0)))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((((data["3__skewness_y"]) + (((((-3.0) + (data["3__skewness_y"]))) + (data["flux_err_std"]))))/2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["4__skewness_y"])), ((data["4__skewness_y"]))))), ((((data["3__kurtosis_y"]) + (data["flux_err_mean"])))))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -3.0, ((np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -3.0, data["flux_err_std"] )) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (3.141593)))) )) +
                0.100000*np.tanh(np.minimum(((data["3__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((((np.minimum(((((data["3__kurtosis_y"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["3__kurtosis_y"])))) + (data["detected_flux_err_median"]))))))))) +
                0.100000*np.tanh(((((data["3__kurtosis_y"]) + (-2.0))) + (((np.minimum(((data["flux_err_std"])), ((data["flux_err_std"])))) + (-2.0))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((-3.0)), ((-3.0))))))))) +
                0.100000*np.tanh(((data["flux_err_std"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))), ((np.maximum(((data["detected_mean"])), ((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (np.where(data["detected_mean"] > -1, -3.0, ((data["flux_std"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, 3.141593, np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], 3.141593 ) )))) +
                0.100000*np.tanh(np.minimum(((data["flux_err_std"])), ((((np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((np.minimum(((data["3__skewness_y"])), ((data["flux_err_std"])))))))), ((data["3__skewness_y"])))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(((np.where(data["detected_mean"] > -1, data["detected_mean"], np.where(data["detected_flux_err_max"]>0, data["flux_err_std"], data["detected_flux_err_mean"] ) )) - (2.0))) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (((data["flux_max"]) * (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((((((-2.0) + (((-2.0) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["flux_err_std"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_mean"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["flux_err_std"]))))), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], 3.141593 )))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_err_std"]))/2.0)) - (data["flux_err_std"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (2.0))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_err_std"]))) + (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((np.minimum(((data["2__kurtosis_y"])), (((((((data["3__kurtosis_y"]) + (data["2__kurtosis_y"]))/2.0)) + (data["flux_d1_pb1"])))))) + (data["flux_diff"]))) +
                0.100000*np.tanh(np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (3.0)))), ((data["detected_mjd_diff"])))) +
                0.100000*np.tanh(np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((data["2__kurtosis_y"])), ((((np.minimum(((data["flux_max"])), ((((data["4__skewness_y"]) * 2.0))))) * 2.0)))))))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_err_std"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"] > -1, data["flux_d0_pb1"], data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb1"])), ((((data["flux_max"]) - (data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((np.where(data["flux_err_max"]>0, ((data["flux_max"]) - (data["detected_flux_ratio_sq_skew"])), ((data["flux_max"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"])) )) - ((-1.0*((data["flux_err_median"])))))) +
                0.100000*np.tanh(((-2.0) + (np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, -2.0, data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["detected_mean"] )))) +
                0.100000*np.tanh(((((data["2__skewness_x"]) - (data["detected_flux_ratio_sq_skew"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((((data["flux_err_mean"]) - (data["3__kurtosis_y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_max"])))) +
                0.100000*np.tanh(np.where(data["flux_max"]<0, data["3__kurtosis_y"], np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["3__kurtosis_y"]))) )) +
                0.100000*np.tanh((((data["4__skewness_y"]) + (np.where(data["3__skewness_x"]>0, np.where(-1.0>0, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ), data["flux_d0_pb3"] )))/2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_max"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((((((-1.0*((data["detected_flux_ratio_sq_skew"])))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((np.where(data["0__skewness_x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["0__skewness_x"] ), np.where(data["0__skewness_x"]>0, data["0__skewness_x"], data["0__skewness_x"] ) ))))) +
                0.100000*np.tanh(np.where(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * (data["0__skewness_x"]))<0, ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0), data["0__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((((((np.minimum(((data["flux_d0_pb1"])), ((data["flux_err_mean"])))) + (-2.0))) + (data["detected_mean"])))), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)<0, data["3__kurtosis_y"], ((data["1__kurtosis_x"]) * 2.0) )))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_d0_pb1"]<0, data["flux_d0_pb1"], np.where(data["flux_d0_pb1"]>0, data["detected_mean"], data["flux_d0_pb1"] ) ))), ((np.minimum(((data["detected_mean"])), ((data["2__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((np.minimum(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["flux_std"]) - (data["5__fft_coefficient__coeff_0__attr__abs__x"])))))), ((((-3.0) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])))))) + (data["flux_std"]))) +
                0.100000*np.tanh(np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh((((((data["flux_err_std"]) + (np.tanh((data["flux_d1_pb0"]))))/2.0)) - (((data["flux_d1_pb2"]) * (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["0__kurtosis_x"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) - (data["flux_err_skew"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((((data["flux_d1_pb1"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))))), ((data["detected_mean"])))) +
                0.100000*np.tanh(((((((((((((data["flux_err_median"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.maximum((((((np.where(data["3__kurtosis_y"]>0, data["3__skewness_y"], data["3__kurtosis_y"] )) + ((((data["3__kurtosis_y"]) + (data["detected_flux_diff"]))/2.0)))/2.0))), ((np.tanh((data["detected_mean"])))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_min"]<0, data["3__kurtosis_y"], np.minimum(((3.141593)), ((np.minimum(((-2.0)), (((((data["2__kurtosis_x"]) + (data["1__skewness_y"]))/2.0))))))) )) / 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb1"])), (((((((-1.0*((np.tanh((data["detected_flux_by_flux_ratio_sq_sum"])))))) / 2.0)) / 2.0))))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"] > -1, ((np.where(data["3__kurtosis_y"]>0, data["flux_d0_pb1"], np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, data["0__skewness_y"], data["flux_d0_pb1"] ) )) * 2.0), data["flux_err_std"] )) +
                0.100000*np.tanh((((np.tanh(((((data["ddf"]) + (data["2__skewness_x"]))/2.0)))) + (data["detected_flux_err_max"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__skewness_y"])), ((np.minimum(((data["flux_err_max"])), ((data["3__kurtosis_y"])))))))), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh((-1.0*((np.where(data["0__kurtosis_y"] > -1, ((data["0__skewness_x"]) - (((data["detected_flux_max"]) - (data["0__skewness_x"])))), data["2__fft_coefficient__coeff_1__attr__abs__x"] ))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_max"])), ((data["3__kurtosis_y"]))))), ((((np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_mean"])))) / 2.0))))) +
                0.100000*np.tanh((((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_mjd_diff"]) / 2.0)))/2.0)) * (np.minimum(((data["flux_err_max"])), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(np.minimum(((((-3.0) + (data["5__fft_coefficient__coeff_0__attr__abs__x"])))), ((((np.where(-3.0>0, data["2__skewness_x"], data["mjd_diff"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["flux_ratio_sq_sum"]))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.minimum(((((data["1__kurtosis_y"]) + (((data["mjd_diff"]) - (((data["2__skewness_y"]) / 2.0))))))), ((data["detected_flux_max"])))) +
                0.100000*np.tanh(np.minimum(((((data["2__kurtosis_x"]) + (data["3__kurtosis_y"])))), ((data["3__kurtosis_y"])))) +
                0.100000*np.tanh(np.minimum(((((((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) * (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) * (((data["4__skewness_y"]) * 2.0))))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((((((3.141593) < (np.where(data["ddf"]<0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )))*1.)) / 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__skewness_y"]))))), ((((((data["flux_ratio_sq_sum"]) / 2.0)) - (data["flux_dif2"])))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_max"])))), ((((data["4__skewness_y"]) * 2.0))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_flux_w_mean"] > -1, data["0__skewness_x"], data["0__skewness_x"] ))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((np.tanh((np.maximum(((np.minimum(((data["2__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["mjd_size"])))))) - (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))))))) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((((data["4__skewness_y"]) / 2.0))), ((((data["flux_d1_pb1"]) - (data["2__skewness_y"])))))) * 2.0))), ((np.minimum(((data["detected_flux_ratio_sq_sum"])), ((data["3__fft_coefficient__coeff_0__attr__abs__x"]))))))) +
                0.100000*np.tanh(((3.0) * ((((data["2__kurtosis_y"]) + (data["detected_mean"]))/2.0)))) +
                0.100000*np.tanh(np.where(np.minimum(((data["4__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["2__kurtosis_y"])), ((np.tanh((data["flux_median"]))))))))>0, ((data["flux_max"]) / 2.0), data["2__skewness_y"] )) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_err_std"])))) +
                0.100000*np.tanh((-1.0*((((((data["2__skewness_x"]) - ((((data["2__kurtosis_y"]) < (data["0__kurtosis_y"]))*1.)))) / 2.0))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), (((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))))), ((data["2__skewness_y"]))))), ((data["2__skewness_x"])))) +
                0.100000*np.tanh(((np.where((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) < (data["detected_flux_dif3"]))*1.) > -1, data["2__fft_coefficient__coeff_1__attr__abs__x"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["detected_flux_err_max"], data["2__kurtosis_y"] )) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["detected_flux_std"]))))), ((((data["hostgal_photoz"]) + ((-1.0*((((3.141593) * (data["detected_flux_err_median"]))))))))))) +
                0.100000*np.tanh(np.where(((((3.67651557922363281)) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) > -1, (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_max"]))/2.0), data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.where(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_y"], np.where(data["1__skewness_y"]>0, data["flux_d1_pb2"], data["flux_d1_pb2"] ) )>0, 2.718282, data["flux_d1_pb2"] )) +
                0.100000*np.tanh(np.where(data["flux_mean"] > -1, data["flux_d1_pb1"], data["2__kurtosis_y"] )) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((((data["2__skewness_x"]) + (((data["3__kurtosis_y"]) * 2.0))))), ((data["3__kurtosis_y"])))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (((2.718282) - (data["detected_flux_by_flux_ratio_sq_sum"]))))))) +
                0.100000*np.tanh(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + ((((data["detected_mjd_diff"]) + (data["detected_flux_dif2"]))/2.0)))) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["flux_d1_pb1"] > -1, data["1__kurtosis_x"], data["flux_median"] )) - (((data["0__skewness_y"]) * (data["2__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["flux_max"]) + ((((np.tanh((((data["flux_err_std"]) + (0.0))))) + (data["flux_median"]))/2.0)))) +
                0.100000*np.tanh(((data["5__skewness_y"]) + (data["1__skewness_y"]))) +
                0.100000*np.tanh((((data["detected_flux_mean"]) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(np.tanh((((data["detected_flux_err_max"]) - ((-1.0*((np.minimum(((((data["detected_flux_err_max"]) / 2.0))), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]))))))))))) +
                0.100000*np.tanh(((np.minimum(((((-3.0) - ((((0.0)) + (2.718282)))))), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["detected_flux_diff"]))) +
                0.100000*np.tanh((((((data["detected_flux_std"]) * (((((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["3__skewness_x"])))) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) + (0.367879))/2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_y"]<0, data["3__kurtosis_y"], (((((np.minimum(((data["3__kurtosis_y"])), ((data["3__kurtosis_y"])))) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) - (((data["flux_err_max"]) * 2.0))) )) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_err_max"])), ((data["detected_mean"])))) +
                0.100000*np.tanh(((np.tanh((data["flux_ratio_sq_sum"]))) - (np.where(data["0__kurtosis_y"]>0, (((data["flux_ratio_sq_skew"]) + (data["4__skewness_y"]))/2.0), data["flux_err_skew"] )))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((((data["flux_mean"]) / 2.0)) - (data["flux_err_std"]))))) +
                0.100000*np.tanh(((data["flux_median"]) - ((((-1.0*((data["5__skewness_x"])))) - (((data["flux_dif2"]) * (data["flux_by_flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(np.where(np.minimum(((data["3__skewness_y"])), ((data["3__fft_coefficient__coeff_1__attr__abs__y"])))<0, np.tanh((((np.tanh((data["3__fft_coefficient__coeff_0__attr__abs__x"]))) / 2.0))), data["1__kurtosis_y"] )) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((((data["2__kurtosis_x"]) / 2.0))), ((data["detected_flux_by_flux_ratio_sq_sum"])))<0, data["flux_d0_pb2"], data["3__kurtosis_y"] ))), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((np.where((((data["flux_d1_pb2"]) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)<0, data["5__skewness_y"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ))), ((data["3__skewness_y"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_diff"] > -1, data["2__kurtosis_y"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["3__kurtosis_y"], data["2__skewness_y"] ) )) +
                0.100000*np.tanh(np.where(data["mwebv"] > -1, data["2__fft_coefficient__coeff_0__attr__abs__x"], data["2__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((data["detected_flux_err_median"])))) +
                0.100000*np.tanh((((((((((np.tanh((data["flux_diff"]))) / 2.0)) * 2.0)) + (data["1__kurtosis_x"]))/2.0)) / 2.0)) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"] > -1, (-1.0*((np.minimum(((data["flux_d0_pb1"])), ((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) / 2.0))))))), -2.0 )))

    def GP_class_62(self,data):
        return (-1.361137 +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["flux_d0_pb5"]) + (((((data["detected_flux_median"]) + (((data["5__skewness_x"]) * 2.0)))) + (data["distmod"])))), -3.0 )) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) + (((((np.maximum(((data["detected_flux_min"])), ((data["detected_flux_min"])))) + (((data["4__kurtosis_x"]) + (data["detected_flux_min"]))))) + (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["2__kurtosis_x"])), ((data["flux_min"]))))), ((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["flux_d1_pb4"]) + (data["5__skewness_x"])))), ((data["5__skewness_x"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((data["flux_d0_pb5"]))))), ((data["flux_d0_pb5"]))))), ((np.minimum(((data["flux_min"])), ((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["5__skewness_x"])))))))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((np.minimum(((data["flux_min"])), ((np.minimum(((data["flux_d0_pb5"])), ((((data["flux_d0_pb5"]) + (data["flux_d0_pb5"])))))))))), ((data["detected_flux_min"]))))), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((1.0)))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_d0_pb1"])), ((np.tanh((data["flux_d0_pb5"]))))))), ((np.minimum(((data["1__skewness_x"])), ((data["1__kurtosis_x"]))))))) +
                0.100000*np.tanh(np.minimum(((data["flux_min"])), ((data["flux_min"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((data["0__kurtosis_x"])))) +
                0.100000*np.tanh(((((((data["distmod"]) - (np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))) - (data["detected_flux_err_std"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((((np.minimum(((data["detected_flux_mean"])), ((data["hostgal_photoz_err"])))) - (((data["hostgal_photoz_err"]) - (-1.0))))))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.minimum((((((((data["detected_flux_min"]) + (np.minimum(((data["5__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))/2.0)) + (data["flux_d0_pb4"])))), ((data["4__skewness_x"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["0__skewness_y"])), ((np.minimum(((data["detected_flux_min"])), ((data["detected_mjd_diff"])))))))), ((data["flux_dif2"])))) +
                0.100000*np.tanh(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), ((((data["flux_dif2"]) * 2.0))))) + (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.minimum(((((data["3__kurtosis_x"]) * 2.0))), (((((((data["5__kurtosis_x"]) < (data["4__skewness_x"]))*1.)) / 2.0))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_min"])), ((((data["detected_flux_min"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(((((data["detected_flux_min"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))))) - (data["flux_err_min"]))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - ((((data["4__skewness_y"]) > (data["flux_dif2"]))*1.)))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(np.minimum(((data["distmod"])), ((data["distmod"]))) > -1, data["flux_d0_pb5"], (((data["distmod"]) + ((((data["distmod"]) + (data["distmod"]))/2.0)))/2.0) )) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], ((data["distmod"]) * 2.0) )) +
                0.100000*np.tanh(((((np.minimum(((((data["hostgal_photoz_err"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))), ((data["detected_flux_ratio_sq_skew"])))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh((((((data["flux_d0_pb5"]) + (data["flux_d1_pb5"]))/2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["5__skewness_x"]) + (data["hostgal_photoz_err"])))), ((data["2__kurtosis_x"]))))), ((((((data["flux_min"]) + (data["2__kurtosis_x"]))) + (0.367879)))))) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(((((((((data["detected_flux_min"]) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["detected_flux_min"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((np.minimum(((((data["2__kurtosis_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])))), ((((data["4__kurtosis_x"]) + ((((1.0) + (data["distmod"]))/2.0))))))) + (data["flux_d0_pb5"]))/2.0)) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_median"])), ((((np.minimum(((((((data["1__kurtosis_x"]) - (data["1__skewness_y"]))) + (data["flux_d0_pb5"])))), ((data["flux_d1_pb5"])))) - (data["1__skewness_y"])))))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (np.maximum(((((data["1__kurtosis_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["hostgal_photoz"])))))) - (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["detected_mjd_diff"]) - (data["5__skewness_x"]))))) - (data["detected_mjd_diff"]))) - (data["flux_max"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["3__skewness_x"]) - (((((data["detected_flux_err_std"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__kurtosis_y"]))))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(3.141593 > -1, data["hostgal_photoz_err"], np.minimum(((data["detected_flux_min"])), (((0.28365617990493774)))) )) +
                0.100000*np.tanh(((data["1__kurtosis_x"]) + (((data["flux_d1_pb2"]) * 2.0)))) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) - (np.where(data["flux_d0_pb2"]<0, data["1__skewness_y"], data["2__kurtosis_x"] )))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((data["mjd_size"]) + (((data["distmod"]) - (((data["detected_flux_by_flux_ratio_sq_sum"]) + (data["1__skewness_y"]))))))) - (data["3__skewness_y"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, data["2__skewness_x"], ((((data["2__skewness_x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh((-1.0*(((-1.0*(((((((-1.0*((data["mjd_diff"])))) + (data["flux_d0_pb4"]))) - (data["2__skewness_y"]))))))))) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["flux_d1_pb0"]))) - (data["detected_mjd_diff"]))) - (((data["flux_d0_pb5"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_mjd_diff"]))) * 2.0)) - (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_flux_dif3"]) - ((((data["flux_d1_pb5"]) < (data["2__fft_coefficient__coeff_0__attr__abs__x"]))*1.)))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["1__kurtosis_x"]) + (data["3__skewness_x"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.where(data["5__skewness_x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_d0_pb5"]) + (data["5__skewness_x"])) )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh((((((data["1__skewness_y"]) < (((data["2__kurtosis_x"]) + (data["detected_mjd_diff"]))))*1.)) - ((((data["detected_mjd_diff"]) > ((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)))*1.)))) +
                0.100000*np.tanh(((((((data["3__kurtosis_x"]) + (data["3__kurtosis_x"]))) + (data["2__kurtosis_x"]))) + (2.718282))) +
                0.100000*np.tanh(np.where(-2.0 > -1, ((data["flux_dif3"]) - (((data["1__skewness_x"]) - (data["detected_flux_by_flux_ratio_sq_skew"])))), ((((data["1__skewness_x"]) - (data["5__kurtosis_y"]))) * 2.0) )) +
                0.100000*np.tanh(((((((((data["detected_flux_dif3"]) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) - ((-1.0*((data["distmod"])))))) +
                0.100000*np.tanh(((((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["5__fft_coefficient__coeff_1__attr__abs__y"] )) - (((2.0) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["1__skewness_y"])))))) < ((((data["1__skewness_y"]) < (data["0__fft_coefficient__coeff_0__attr__abs__y"]))*1.)))*1.)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["5__skewness_x"]) * 2.0)) - (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((np.maximum(((data["mjd_diff"])), ((data["2__skewness_x"])))) - (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) * 2.0)) < (data["flux_d0_pb5"]))*1.)) - (data["flux_d1_pb0"]))) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["flux_ratio_sq_sum"]))) - (((((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(((data["flux_d0_pb0"]) + (((data["2__skewness_x"]) + (((((((np.tanh((data["flux_ratio_sq_sum"]))) / 2.0)) * 2.0)) * 2.0)))))) +
                0.100000*np.tanh((((((data["flux_w_mean"]) < (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_d0_pb5"]) * 2.0) )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (data["1__skewness_y"]))) + (data["1__kurtosis_x"]))) - (data["1__skewness_y"]))) +
                0.100000*np.tanh(((((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) * (data["detected_flux_dif3"]))) + (np.minimum(((data["3__kurtosis_x"])), ((data["3__kurtosis_x"])))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["flux_dif2"]))/2.0)) + (data["flux_dif2"]))/2.0)) +
                0.100000*np.tanh((((((data["flux_d0_pb2"]) < (data["flux_mean"]))*1.)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d0_pb0"])))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_dif3"]) + (((data["detected_flux_dif3"]) + (((data["detected_flux_dif3"]) - (((data["flux_max"]) * 2.0)))))))) +
                0.100000*np.tanh(((((data["0__kurtosis_y"]) + (((data["3__kurtosis_y"]) + (data["detected_flux_err_min"]))))) + ((-1.0*((data["4__skewness_y"])))))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (data["2__skewness_x"]))) + (((((data["2__skewness_x"]) * 2.0)) + (np.where(data["2__skewness_x"]<0, data["2__skewness_x"], data["2__skewness_x"] )))))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_min"] > -1, np.tanh((data["detected_flux_dif3"])), data["detected_flux_dif3"] )) +
                0.100000*np.tanh(np.where(np.where(np.where(data["4__kurtosis_x"]<0, data["detected_flux_dif3"], data["flux_d0_pb0"] )<0, data["flux_d0_pb0"], data["mwebv"] ) > -1, data["flux_d0_pb0"], ((data["flux_d0_pb0"]) * 2.0) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, ((data["flux_d0_pb5"]) + (data["detected_flux_err_max"])), np.where(data["5__skewness_x"]<0, data["detected_flux_dif3"], ((data["detected_flux_err_min"]) + (data["0__kurtosis_y"])) ) )) +
                0.100000*np.tanh(((np.where(((data["mjd_size"]) - (((data["mwebv"]) - (data["5__kurtosis_y"]))))<0, data["3__kurtosis_y"], data["3__kurtosis_y"] )) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["1__skewness_x"])), ((((data["detected_flux_dif3"]) * (data["detected_flux_dif3"])))))) * (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((np.where(((data["flux_d0_pb1"]) - (data["flux_median"])) > -1, data["flux_d1_pb1"], data["flux_d0_pb1"] )) > (data["flux_d0_pb1"]))*1.)) - (data["flux_median"]))) +
                0.100000*np.tanh(((((((((((((data["detected_flux_dif3"]) + (data["flux_dif3"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (data["flux_dif3"]))) +
                0.100000*np.tanh(((((((np.where(((data["detected_flux_max"]) * 2.0)>0, ((data["flux_d1_pb4"]) - (data["flux_d0_pb1"])), data["detected_flux_err_min"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_d1_pb2"])))) +
                0.100000*np.tanh((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )))*1.)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, np.where(data["1__skewness_x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["3__kurtosis_x"] ), data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((((((data["detected_flux_dif3"]) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["flux_median"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_err_min"]) + (np.maximum(((((((data["detected_flux_err_min"]) + (data["hostgal_photoz"]))) + (np.maximum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_flux_err_min"]))))))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((data["flux_diff"]) < (data["1__skewness_x"]))*1.), data["detected_flux_by_flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb3"]>0, data["4__kurtosis_x"], np.where(data["flux_d1_pb4"]>0, np.where(data["flux_d1_pb3"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["flux_err_skew"] ), data["4__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(np.where(data["distmod"]<0, data["distmod"], data["mwebv"] )<0, ((data["detected_flux_dif3"]) + (data["0__kurtosis_y"])), data["distmod"] )) +
                0.100000*np.tanh((((data["distmod"]) + (np.where(data["hostgal_photoz"]>0, (((data["distmod"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.), data["mwebv"] )))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]<0, data["detected_flux_err_median"], data["flux_max"] )) +
                0.100000*np.tanh(((((data["0__skewness_x"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) * (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["2__skewness_x"] > -1, data["detected_flux_dif3"], data["detected_mjd_size"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_dif3"] > -1, ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["detected_flux_ratio_sq_skew"])), ((np.where(data["flux_mean"]<0, data["flux_ratio_sq_skew"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) * (data["3__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"]>0, (((data["mjd_size"]) < (data["hostgal_photoz"]))*1.), (((data["mwebv"]) + (np.minimum(((data["hostgal_photoz_err"])), ((data["1__skewness_x"])))))/2.0) )) +
                0.100000*np.tanh(((((((data["flux_d1_pb3"]) * 2.0)) * (np.where(np.where(data["flux_d1_pb3"] > -1, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb4"] ) > -1, data["4__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb3"] )))) * 2.0)) +
                0.100000*np.tanh(np.maximum(((data["flux_d0_pb0"])), ((((data["flux_d0_pb0"]) * 2.0))))) +
                0.100000*np.tanh(((((((((data["0__kurtosis_x"]) - (data["detected_mean"]))) - (((data["mjd_size"]) * 2.0)))) - (((data["0__kurtosis_x"]) * 2.0)))) - (data["5__kurtosis_y"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["flux_d0_pb0"], (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) > (np.where((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (data["3__kurtosis_x"]))*1.) > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["3__kurtosis_x"] )))*1.) )) +
                0.100000*np.tanh(((((((((((np.where(data["detected_mean"]>0, data["detected_flux_median"], data["2__skewness_y"] )) - (data["flux_median"]))) * 2.0)) * 2.0)) - (data["flux_median"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["flux_d0_pb5"]<0, ((data["distmod"]) + (data["3__fft_coefficient__coeff_0__attr__abs__x"])), data["detected_flux_dif3"] ) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_median"]))) + (((data["detected_flux_err_median"]) + (((data["2__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["distmod"] > -1, (-1.0*((data["flux_median"]))), data["distmod"] ), ((data["hostgal_photoz_err"]) + (data["flux_median"])) )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb0"]<0, data["2__skewness_x"], np.where(data["flux_d0_pb0"]<0, data["flux_ratio_sq_skew"], (7.80123519897460938) ) )) * 2.0)) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) * (data["4__kurtosis_x"]))) * (((data["flux_d0_pb5"]) * (data["1__skewness_x"]))))) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))<0, ((data["mwebv"]) - (data["3__kurtosis_x"])), np.where(data["distmod"]<0, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__x"] ) )) +
                0.100000*np.tanh(((((data["detected_flux_dif3"]) - (data["flux_median"]))) + (((data["detected_flux_dif3"]) - (((data["detected_flux_dif3"]) * (((data["detected_flux_dif3"]) - (data["flux_median"]))))))))) +
                0.100000*np.tanh(((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], ((((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, np.where(data["hostgal_photoz"]<0, np.where(data["detected_flux_err_skew"]<0, data["hostgal_photoz_err"], data["hostgal_photoz"] ), data["hostgal_photoz_err"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(np.where(np.where(data["flux_skew"]>0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["flux_skew"] )>0, data["distmod"], ((data["flux_skew"]) + (data["0__skewness_x"])) )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]<0, data["detected_flux_err_min"], data["3__kurtosis_y"] )) +
                0.100000*np.tanh(((((np.where(data["hostgal_photoz_err"]>0, data["distmod"], ((((data["hostgal_photoz_err"]) + (data["flux_d1_pb4"]))) + (data["flux_d1_pb5"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_median"]>0, data["flux_err_min"], ((np.where(data["detected_flux_err_median"]>0, data["flux_err_min"], data["2__skewness_x"] )) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.) )) +
                0.100000*np.tanh(((data["detected_flux_err_max"]) - (data["0__skewness_y"]))) +
                0.100000*np.tanh((((((data["detected_flux_err_std"]) < ((((data["detected_mjd_size"]) > (((data["flux_by_flux_ratio_sq_sum"]) - (np.maximum(((data["flux_mean"])), ((data["detected_flux_err_skew"])))))))*1.)))*1.)) - (data["detected_flux_err_std"]))) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) < (((((((data["detected_mjd_size"]) < ((((data["2__kurtosis_y"]) < ((((data["detected_mjd_diff"]) < (data["detected_flux_min"]))*1.)))*1.)))*1.)) < (data["4__fft_coefficient__coeff_0__attr__abs__x"]))*1.)))*1.)) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) - (data["1__skewness_x"])))), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["flux_dif2"]>0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__kurtosis_y"], data["0__fft_coefficient__coeff_1__attr__abs__x"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) > (data["1__fft_coefficient__coeff_1__attr__abs__x"]))*1.), data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["3__skewness_x"])))) +
                0.100000*np.tanh(np.where(np.where(data["0__skewness_x"] > -1, (((data["flux_d0_pb2"]) < (data["flux_d0_pb2"]))*1.), data["flux_d0_pb2"] )<0, data["detected_mjd_size"], (((data["flux_d0_pb2"]) < (data["flux_mean"]))*1.) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_diff"])), ((data["2__skewness_x"]))))), ((np.minimum(((np.minimum(((data["flux_d1_pb1"])), ((data["distmod"]))))), ((data["flux_d1_pb1"]))))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, (((data["5__skewness_x"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.), data["detected_mjd_size"] )) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"]>0, np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_err_skew"], np.where(data["detected_flux_err_median"]>0, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ) ), data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.where(data["hostgal_photoz_err"]<0, np.where(data["flux_d0_pb4"]<0, data["hostgal_photoz_err"], data["detected_mjd_size"] ), data["detected_mjd_size"] ), data["hostgal_photoz_err"] )) * 2.0)) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) < ((((((data["detected_mjd_diff"]) < (data["detected_mjd_diff"]))*1.)) * 2.0)))*1.)) +
                0.100000*np.tanh(np.where(((data["flux_dif2"]) * (data["flux_dif2"]))>0, data["1__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] ) )))

    def GP_class_64(self,data):
        return (-2.164979 +
                0.100000*np.tanh(((((((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) + (((((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) * 2.0)))) - (((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((-1.0*((((data["detected_mjd_diff"]) * 2.0))))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["flux_err_max"]) - (np.where(((((data["detected_mjd_diff"]) * 2.0)) * 2.0) > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) - (np.tanh((data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))/2.0)) + ((-1.0*((data["detected_mjd_size"])))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) * 2.0)))) +
                0.100000*np.tanh((((((-1.0*((np.where((-1.0*((data["detected_mjd_diff"])))>0, data["detected_mjd_diff"], data["detected_mjd_diff"] ))))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((((((((np.tanh((data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) / 2.0)))) +
                0.100000*np.tanh(((((np.maximum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))) * 2.0)) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - (((((data["detected_mjd_diff"]) * 2.0)) * 2.0)))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) > (data["1__skewness_y"]))*1.)) - ((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))/2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.where(((data["flux_ratio_sq_skew"]) * 2.0) > -1, (-1.0*((data["detected_mjd_diff"]))), ((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"])) )) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) * 2.0)))) + (data["flux_ratio_sq_skew"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.minimum((((-1.0*((data["detected_mjd_diff"]))))), (((((((-1.0*((data["detected_mjd_diff"])))) * 2.0)) + (data["flux_ratio_sq_skew"])))))) * 2.0)) +
                0.100000*np.tanh(((((((data["detected_flux_mean"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_ratio_sq_skew"]) - (data["detected_mjd_size"]))) - (((data["detected_mjd_size"]) - (data["flux_d0_pb1"]))))) - (data["detected_mjd_size"]))) * 2.0)) +
                0.100000*np.tanh((-1.0*((((1.0) + (((data["detected_mjd_diff"]) + (data["detected_mjd_diff"])))))))) +
                0.100000*np.tanh((-1.0*((((((((1.0) + (data["detected_mjd_diff"]))) * 2.0)) + (((((data["detected_mjd_diff"]) * 2.0)) + (1.0)))))))) +
                0.100000*np.tanh((-1.0*((((((((data["detected_mjd_diff"]) * 2.0)) * 2.0)) - (((data["detected_mjd_diff"]) * 2.0))))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) * 2.0)))) + (((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_sum"]))) * 2.0)))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_ratio_sq_skew"])), ((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"])))))) + (data["flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_mjd_size"]))) - (data["detected_mjd_size"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((-1.0*((np.where((((((((data["detected_mjd_diff"]) / 2.0)) + (data["flux_median"]))/2.0)) * 2.0)<0, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) * 2.0) ))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_mjd_diff"] > -1, np.where(((data["detected_mjd_diff"]) / 2.0) > -1, 2.718282, data["detected_mjd_diff"] ), data["detected_mjd_diff"] ))))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, -3.0, np.where(data["detected_mjd_diff"] > -1, -3.0, (8.0) ) )) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) - (data["flux_ratio_sq_skew"]))) * 2.0)) + (((np.maximum(((data["detected_flux_std"])), ((data["detected_flux_std"])))) + (data["flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((data["detected_flux_std"]) - (data["detected_flux_diff"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))) + (((data["detected_flux_dif2"]) * 2.0)))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh(((((((-1.0) * 2.0)) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_std"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) - (data["flux_d0_pb4"]))) - (data["ddf"]))) +
                0.100000*np.tanh(((((((data["detected_flux_std"]) - (data["flux_d0_pb4"]))) - (data["flux_d0_pb4"]))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_max"]) - (data["flux_d0_pb5"]))) - (((data["flux_d0_pb5"]) + (data["detected_mjd_diff"]))))) - (data["flux_median"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["flux_err_mean"]) + (((((((data["detected_flux_std"]) - (data["detected_mjd_diff"]))) - (data["flux_median"]))) - (data["flux_median"]))))) - (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((np.minimum(((data["detected_flux_std"])), ((((data["flux_dif2"]) + (data["detected_flux_mean"])))))) - (data["flux_d0_pb5"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) + (((data["detected_flux_std"]) + (-2.0))))) + (((data["detected_flux_std"]) - (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh(np.where((-1.0*((data["hostgal_photoz_err"]))) > -1, np.where((2.44751977920532227) > -1, np.where(data["detected_mjd_diff"] > -1, -3.0, 3.0 ), 3.0 ), 3.0 )) +
                0.100000*np.tanh((((-1.0*((((data["flux_d0_pb5"]) + (((data["flux_d0_pb5"]) + (data["flux_d0_pb4"])))))))) - (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, -2.0, (-1.0*((-2.0))) )) +
                0.100000*np.tanh(((((data["detected_flux_std"]) + (((data["flux_err_max"]) + (np.tanh(((-1.0*((data["flux_dif3"])))))))))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_std"]) * 2.0)) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_err_mean"]) - (np.where(data["detected_mjd_diff"]<0, data["detected_mjd_diff"], data["flux_err_mean"] )))) - (np.where(data["flux_err_mean"]<0, data["detected_mjd_diff"], (8.0) )))) +
                0.100000*np.tanh((((((data["hostgal_photoz"]) + (data["flux_dif2"]))/2.0)) + (((data["detected_flux_diff"]) + (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"] > -1, -3.0, data["detected_flux_median"] )) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_diff"]) - (data["0__kurtosis_x"]))) - (data["1__kurtosis_x"]))) - (data["0__kurtosis_x"]))) - (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((np.minimum(((data["detected_flux_diff"])), ((data["detected_flux_diff"])))) - (data["1__skewness_x"]))) - (data["1__skewness_x"]))) - ((2.0)))) +
                0.100000*np.tanh(((((((data["detected_flux_diff"]) - (((1.0) - (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))))) - (data["mjd_size"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((data["detected_flux_std"]) - (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["detected_flux_std"] > -1, data["5__skewness_x"], data["detected_flux_std"] ), ((data["3__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_std"])) )))) +
                0.100000*np.tanh(np.where(((data["hostgal_photoz"]) + (data["flux_d0_pb4"])) > -1, ((data["hostgal_photoz"]) - (3.141593)), (-1.0*((data["hostgal_photoz"]))) )) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_std"])), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] )) + (np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_err_skew"] )))) +
                0.100000*np.tanh(((((((((np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz"])))) * 2.0)) * (((data["detected_mjd_diff"]) + (data["hostgal_photoz"]))))) - (2.718282))) * 2.0)) +
                0.100000*np.tanh(np.where(((data["flux_skew"]) * 2.0) > -1, ((data["hostgal_photoz"]) + (-3.0)), data["detected_flux_std"] )) +
                0.100000*np.tanh(((np.where(data["detected_flux_std"] > -1, ((data["detected_mean"]) - (data["flux_d0_pb5"])), data["flux_w_mean"] )) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["flux_err_mean"]>0, -2.0, ((data["flux_err_mean"]) - (-2.0)) )) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, data["flux_ratio_sq_skew"], ((data["detected_mjd_diff"]) * (-3.0)) )) +
                0.100000*np.tanh(((np.where(data["0__kurtosis_x"]<0, data["4__kurtosis_x"], ((data["hostgal_photoz"]) - (np.where(data["0__kurtosis_x"]<0, data["detected_flux_skew"], data["0__kurtosis_x"] ))) )) + (data["flux_d1_pb3"]))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]>0, ((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (np.minimum(((-3.0)), ((data["2__skewness_y"]))))), data["detected_flux_mean"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (data["distmod"])) > -1, ((data["hostgal_photoz"]) * 2.0), ((((data["hostgal_photoz"]) * 2.0)) + (3.0)) )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((-2.0) + (data["hostgal_photoz"])), data["flux_w_mean"] )) +
                0.100000*np.tanh(((((np.minimum(((((((data["detected_flux_std"]) - (data["flux_d1_pb5"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["detected_flux_w_mean"])))) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_skew"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__skewness_y"]))))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_std"]) + (((((data["2__skewness_x"]) + (data["2__kurtosis_x"]))) + (data["detected_flux_std"]))))) +
                0.100000*np.tanh((((data["hostgal_photoz"]) + (((((data["hostgal_photoz"]) - (2.0))) + (((data["hostgal_photoz"]) * (((data["hostgal_photoz"]) - ((3.13035678863525391)))))))))/2.0)) +
                0.100000*np.tanh((((((-3.0) + (((data["hostgal_photoz"]) + (data["detected_flux_diff"]))))/2.0)) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((np.where(data["flux_skew"]<0, data["detected_flux_median"], data["flux_max"] )) + (data["flux_dif2"]))) +
                0.100000*np.tanh(((((-1.0) - (data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) - (data["detected_flux_max"]))))) +
                0.100000*np.tanh((-1.0*(((-1.0*(((-1.0*((np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["mjd_diff"], data["detected_flux_err_skew"] ))))))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_max"]>0, np.where(data["flux_d1_pb4"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_0__attr__abs__y"] ), -2.0 )) +
                0.100000*np.tanh(np.where(((((1.0) - (data["flux_d0_pb5"]))) + (data["detected_flux_std"])) > -1, data["detected_flux_std"], data["ddf"] )) +
                0.100000*np.tanh(np.where(data["flux_err_std"]<0, data["detected_flux_std"], ((((data["flux_err_std"]) - (data["flux_err_std"]))) - (((data["detected_flux_std"]) + (data["detected_flux_std"])))) )) +
                0.100000*np.tanh((-1.0*((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + ((((data["flux_d0_pb5"]) > ((-1.0*((data["mjd_size"])))))*1.))))))))) +
                0.100000*np.tanh(((data["flux_d1_pb0"]) + (((((data["flux_w_mean"]) - (data["0__kurtosis_x"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) - (data["distmod"]))) * 2.0)) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) * 2.0)) * (np.where((1.0)<0, ((data["distmod"]) * 2.0), ((data["distmod"]) * 2.0) )))) + (-2.0))) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["flux_d1_pb4"], (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))/2.0) ), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(np.minimum(((data["2__skewness_x"])), ((np.minimum(((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["2__skewness_x"]))))))) +
                0.100000*np.tanh(np.maximum(((data["2__skewness_x"])), ((((data["2__skewness_x"]) / 2.0))))) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, np.where(data["flux_w_mean"] > -1, np.where(data["flux_w_mean"]<0, 2.0, data["detected_flux_max"] ), data["flux_w_mean"] ), -2.0 )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, np.where(data["detected_mjd_diff"]>0, -2.0, -2.0 ), data["4__kurtosis_x"] )) +
                0.100000*np.tanh((((data["detected_flux_ratio_sq_skew"]) + (np.maximum((((((data["detected_flux_w_mean"]) + (data["flux_d1_pb3"]))/2.0))), ((np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))))))))/2.0)) +
                0.100000*np.tanh(np.tanh((((((np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]>0, ((data["detected_flux_std"]) - (data["detected_flux_dif2"])), data["flux_w_mean"] )) * 2.0)) - (data["1__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(np.where(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["2__skewness_x"] )<0, data["hostgal_photoz"], data["flux_by_flux_ratio_sq_skew"] )<0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["mjd_diff"] )) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["detected_flux_err_skew"]))) + (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.where(data["flux_ratio_sq_skew"] > -1, np.where(np.maximum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_std"]))) > -1, data["detected_flux_skew"], data["detected_flux_err_std"] ), data["flux_dif2"] )) +
                0.100000*np.tanh((((-1.0*((((data["3__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))) - (((((data["hostgal_photoz_err"]) - ((((data["flux_diff"]) > (data["distmod"]))*1.)))) * (data["3__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_flux_err_mean"], np.where(data["detected_flux_err_max"]>0, data["3__fft_coefficient__coeff_0__attr__abs__y"], data["2__kurtosis_x"] ) ) )) +
                0.100000*np.tanh(((((data["detected_flux_dif2"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["ddf"]) - (data["detected_mjd_diff"]))) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["ddf"]))) +
                0.100000*np.tanh((((np.minimum(((data["flux_std"])), ((np.tanh((data["flux_diff"])))))) + (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_d0_pb5"]))))/2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_max"])), ((((data["5__skewness_y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_0__attr__abs__y"]<0, data["3__fft_coefficient__coeff_0__attr__abs__y"], np.where(data["5__kurtosis_x"]<0, np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["2__skewness_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] ), data["3__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["flux_d1_pb4"] > -1, data["flux_d1_pb4"], np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]<0, data["flux_d1_pb4"], data["flux_ratio_sq_skew"] ) ), data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["detected_mjd_diff"], np.where(data["hostgal_photoz_err"] > -1, data["flux_diff"], data["detected_flux_err_mean"] ) )) +
                0.100000*np.tanh(((data["hostgal_photoz"]) - (3.0))) +
                0.100000*np.tanh(np.where(data["0__kurtosis_x"]<0, data["4__kurtosis_x"], data["2__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(np.where(data["2__kurtosis_x"]<0, data["3__skewness_x"], data["2__skewness_x"] ) > -1, data["2__kurtosis_x"], np.where(data["2__kurtosis_x"] > -1, data["2__skewness_x"], data["3__skewness_x"] ) )) +
                0.100000*np.tanh(((((np.minimum(((((data["ddf"]) - (((data["flux_err_max"]) - (data["ddf"])))))), ((data["0__skewness_x"])))) - (data["ddf"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((np.minimum(((-2.0)), ((data["detected_mjd_diff"])))) - (data["distmod"]))) - (((data["distmod"]) * 2.0)))) * 2.0)) - (data["distmod"]))) +
                0.100000*np.tanh((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["hostgal_photoz"]))/2.0)) + (((data["0__kurtosis_y"]) + (((data["detected_flux_dif2"]) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh((-1.0*((np.where(data["detected_mjd_diff"]<0, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, data["detected_mjd_diff"], data["3__kurtosis_y"] ), np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__kurtosis_x"], data["detected_mjd_diff"] ) ))))) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_sum"]<0, ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["mjd_diff"]))) - (data["mjd_diff"]))) / 2.0), ((data["flux_err_std"]) - (data["mjd_diff"])) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["2__kurtosis_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] ) ) )) +
                0.100000*np.tanh(np.where(data["distmod"]<0, data["detected_flux_std"], data["detected_flux_err_mean"] )) +
                0.100000*np.tanh((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) + (data["hostgal_photoz"]))/2.0)) +
                0.100000*np.tanh(((((data["detected_flux_diff"]) / 2.0)) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.tanh((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["5__kurtosis_y"]<0, (-1.0*((data["2__fft_coefficient__coeff_1__attr__abs__x"]))), np.tanh((np.where(((data["flux_dif2"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) > -1, data["flux_dif2"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ))) )) +
                0.100000*np.tanh(np.where(data["4__kurtosis_x"]>0, data["detected_flux_w_mean"], ((data["detected_flux_dif2"]) / 2.0) )) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], ((data["2__kurtosis_x"]) + (data["detected_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["mjd_size"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))), ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_dif2"])))))) +
                0.100000*np.tanh(((((-1.0*((data["flux_ratio_sq_sum"])))) + ((-1.0*((data["flux_ratio_sq_sum"])))))/2.0)) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["3__fft_coefficient__coeff_1__attr__abs__y"]))>0, np.where(data["flux_d1_pb1"]<0, data["flux_err_mean"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, data["3__fft_coefficient__coeff_0__attr__abs__y"], (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["4__skewness_y"]))/2.0) )) +
                0.100000*np.tanh(((data["hostgal_photoz"]) - (((((((((((((data["distmod"]) * 2.0)) - (data["distmod"]))) * 2.0)) - (data["hostgal_photoz"]))) * 2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((data["distmod"]) * (np.where(data["3__fft_coefficient__coeff_0__attr__abs__x"]>0, data["flux_d1_pb4"], data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) * 2.0)) * 2.0)))

    def GP_class_65(self,data):
        return (-0.972955 +
                0.100000*np.tanh((((-1.0*((((data["distmod"]) + (((data["distmod"]) + (((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) + (2.0)))))))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["detected_mjd_diff"])), ((((np.minimum(((data["flux_ratio_sq_skew"])), ((((np.minimum(((data["detected_mjd_diff"])), ((data["flux_ratio_sq_skew"])))) * 2.0))))) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -3.0, np.where(data["distmod"] > -1, data["distmod"], ((data["flux_by_flux_ratio_sq_skew"]) * 2.0) ) )) +
                0.100000*np.tanh(((((-2.0) + (data["flux_ratio_sq_skew"]))) + (((((data["detected_mjd_size"]) - (data["detected_mjd_size"]))) - (data["distmod"]))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)))) + (((np.minimum(((data["detected_mjd_diff"])), ((data["flux_by_flux_ratio_sq_skew"])))) - ((2.0)))))) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["detected_mjd_diff"], ((((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) - (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_ratio_sq_skew"]))))) - (data["detected_mjd_size"]))) +
                0.100000*np.tanh(((((((data["flux_by_flux_ratio_sq_skew"]) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((((data["detected_mjd_diff"]) * 2.0))), ((((((((((data["flux_ratio_sq_skew"]) - (data["detected_mjd_size"]))) * 2.0)) * 2.0)) - (data["detected_mjd_size"])))))) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz"] > -1, data["flux_ratio_sq_skew"], data["hostgal_photoz"] ) > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, -2.0, data["hostgal_photoz"] ), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.minimum(((np.where(((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))<0, -3.0, data["detected_mjd_diff"] ))), ((data["detected_mjd_diff"])))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - ((((((-3.0) < (data["4__fft_coefficient__coeff_1__attr__abs__y"]))*1.)) * 2.0)))) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.minimum((((((((data["flux_by_flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) + (-3.0))/2.0))), ((data["flux_by_flux_ratio_sq_skew"])))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_ratio_sq_skew"]) + (data["detected_mjd_diff"]))) - (2.0))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_ratio_sq_skew"])), ((((data["detected_flux_err_mean"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"])))))) + (np.minimum(((data["flux_ratio_sq_skew"])), ((data["2__kurtosis_x"])))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) + (((((data["flux_by_flux_ratio_sq_skew"]) * (data["detected_mjd_diff"]))) + (((-2.0) - (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) - (((((((data["distmod"]) + ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))/2.0)) + (3.141593))/2.0)))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_skew"]) + (((((((-2.0) - (data["distmod"]))) - (data["distmod"]))) + (((-3.0) + (data["flux_ratio_sq_skew"]))))))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, np.where(data["flux_by_flux_ratio_sq_skew"] > -1, data["hostgal_photoz"], np.where(data["hostgal_photoz"] > -1, -2.0, data["flux_by_flux_ratio_sq_skew"] ) ), data["flux_by_flux_ratio_sq_skew"] )) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((data["flux_by_flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["detected_mjd_diff"])))) * 2.0)) + (np.tanh((((data["flux_ratio_sq_skew"]) * (data["detected_mjd_diff"]))))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) - (1.0))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["2__kurtosis_y"] )) - (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((-2.0) - (data["flux_dif3"])), np.where(data["hostgal_photoz"] > -1, ((data["hostgal_photoz"]) - (data["hostgal_photoz"])), data["flux_by_flux_ratio_sq_skew"] ) )) +
                0.100000*np.tanh((((((data["flux_ratio_sq_skew"]) + (data["1__skewness_x"]))/2.0)) + ((((data["flux_ratio_sq_skew"]) + (((data["flux_skew"]) * 2.0)))/2.0)))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, np.where(-3.0 > -1, data["hostgal_photoz"], -3.0 ), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, -1.0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) + (-3.0))) + (np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((((np.minimum(((data["2__kurtosis_y"])), ((((-2.0) + (data["detected_mjd_diff"])))))) + (data["detected_mjd_diff"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["1__skewness_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["4__kurtosis_x"]) + (((data["flux_skew"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((data["flux_skew"]) - (data["distmod"]))) > (data["detected_flux_by_flux_ratio_sq_skew"]))*1.)) - (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(np.where(-3.0<0, np.where(data["distmod"] > -1, -3.0, data["detected_mjd_diff"] ), np.where(data["detected_mjd_diff"] > -1, data["distmod"], data["flux_ratio_sq_skew"] ) )) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["detected_mjd_diff"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) - (np.maximum(((data["2__skewness_y"])), (((((((data["2__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) / 2.0))))))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) - (((data["distmod"]) - (((-2.0) - (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__skewness_x"]))) * 2.0)))) +
                0.100000*np.tanh(np.where(data["flux_skew"]>0, ((data["ddf"]) + (data["2__skewness_x"])), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, (((6.0)) - ((7.0))), (6.0) )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((((np.minimum(((-1.0)), ((-1.0)))) + (data["detected_mjd_diff"]))) + (np.minimum(((0.0)), ((data["flux_by_flux_ratio_sq_sum"])))))))) +
                0.100000*np.tanh(np.where(np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))>0, np.where(data["flux_ratio_sq_skew"] > -1, data["flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ), data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"]<0, np.where(data["hostgal_photoz"] > -1, np.where(data["flux_skew"] > -1, -3.0, data["flux_skew"] ), data["flux_skew"] ), data["flux_skew"] )) +
                0.100000*np.tanh((((-1.0*((data["distmod"])))) + (np.where(data["flux_median"]>0, -3.0, (((-3.0) + (data["3__kurtosis_x"]))/2.0) )))) +
                0.100000*np.tanh(((((data["3__kurtosis_x"]) - (((data["flux_d0_pb3"]) - (data["flux_skew"]))))) - (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(((data["flux_by_flux_ratio_sq_skew"]) - (-3.0)) > -1, np.where(((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])) > -1, -3.0, data["flux_ratio_sq_skew"] ), data["flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((((-2.0) + (data["distmod"]))) - (((data["distmod"]) - (-2.0))))) - (data["distmod"]))) - (data["distmod"]))) +
                0.100000*np.tanh(((data["1__kurtosis_y"]) + (((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) + (-2.0))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((np.tanh((-2.0))) + (data["detected_mjd_diff"]))))) +
                0.100000*np.tanh((((((data["2__skewness_x"]) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__skewness_x"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) * (data["flux_skew"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) - ((((2.718282) > (data["detected_flux_ratio_sq_skew"]))*1.)))) +
                0.100000*np.tanh(((data["flux_std"]) + (np.where(data["3__kurtosis_x"]>0, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], (((data["3__kurtosis_x"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))/2.0) )) +
                0.100000*np.tanh((((((data["detected_flux_min"]) + (data["flux_ratio_sq_skew"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(-2.0 > -1, ((((-2.0) - (data["distmod"]))) * 2.0), ((((-2.0) - (data["distmod"]))) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(((np.where(data["flux_err_std"] > -1, ((((data["flux_err_std"]) * 2.0)) * 2.0), data["3__skewness_x"] )) * 2.0)) +
                0.100000*np.tanh(((data["2__kurtosis_y"]) + (((data["2__kurtosis_y"]) + (((data["3__skewness_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["flux_median"]) - (data["flux_median"]))))) +
                0.100000*np.tanh(((((-2.0) + (((((np.minimum(((-2.0)), ((-2.0)))) - (data["distmod"]))) - (data["distmod"]))))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["detected_flux_min"]) - (data["flux_mean"]))) * 2.0)) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["distmod"]))) - (2.0))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (((data["5__skewness_x"]) - (((data["distmod"]) + (3.0))))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["flux_by_flux_ratio_sq_skew"]) - (data["flux_median"]))) - (data["flux_median"]))) - (data["flux_std"]))) - (data["flux_median"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["flux_skew"] ), np.where(data["hostgal_photoz"]>0, data["hostgal_photoz"], -3.0 ) )) +
                0.100000*np.tanh(np.where(((data["distmod"]) / 2.0) > -1, -3.0, ((data["distmod"]) + ((5.06640815734863281))) )) +
                0.100000*np.tanh(((((((((data["2__skewness_x"]) + (data["0__skewness_y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["flux_d0_pb1"]))) + (data["2__skewness_x"]))) +
                0.100000*np.tanh((((((((data["detected_flux_min"]) + (data["3__kurtosis_y"]))) + (data["5__kurtosis_y"]))) + (data["flux_skew"]))/2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], np.minimum(((data["flux_skew"])), ((data["0__skewness_x"]))) )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((((((data["flux_err_mean"]) * 2.0)) - (data["mwebv"]))) < (((((data["flux_err_std"]) / 2.0)) * (data["2__skewness_y"]))))*1.)))) +
                0.100000*np.tanh(np.where(data["0__skewness_x"] > -1, data["3__kurtosis_x"], data["3__kurtosis_x"] )) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["flux_ratio_sq_skew"], np.minimum(((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"])))), ((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"]))))) )) +
                0.100000*np.tanh((((np.where((((data["1__kurtosis_y"]) < ((((data["3__kurtosis_x"]) + (data["0__skewness_x"]))/2.0)))*1.)<0, data["detected_flux_ratio_sq_sum"], data["1__kurtosis_x"] )) + (data["1__kurtosis_x"]))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_std"]))))) +
                0.100000*np.tanh(((np.where(((((((((data["flux_err_median"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)>0, data["3__kurtosis_x"], ((data["ddf"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(1.0<0, ((data["distmod"]) - (((-3.0) - (data["distmod"])))), ((-3.0) - (((data["distmod"]) * 2.0))) )) +
                0.100000*np.tanh(((((np.minimum(((-2.0)), ((((-2.0) - (data["distmod"])))))) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(((data["flux_skew"]) + (np.where(data["flux_skew"]<0, data["flux_err_std"], (((data["1__skewness_x"]) + (np.where(data["0__kurtosis_y"] > -1, data["flux_d0_pb1"], data["flux_err_std"] )))/2.0) )))) +
                0.100000*np.tanh(((((((data["flux_by_flux_ratio_sq_skew"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) - (np.minimum((((-1.0*((data["3__skewness_x"]))))), ((((data["4__kurtosis_y"]) * 2.0))))))) +
                0.100000*np.tanh(((((((((data["0__skewness_x"]) + (data["0__skewness_x"]))/2.0)) + (data["mwebv"]))/2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((data["detected_flux_min"]) * 2.0))), ((((((data["flux_ratio_sq_skew"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["detected_flux_by_flux_ratio_sq_skew"])))))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (((data["flux_mean"]) - (data["detected_mjd_diff"]))))) - (data["detected_flux_err_skew"]))) - (1.0))) +
                0.100000*np.tanh(((data["2__skewness_x"]) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.where(data["flux_diff"] > -1, data["4__kurtosis_x"], data["flux_err_mean"] )) +
                0.100000*np.tanh(((((((data["flux_median"]) - (data["flux_median"]))) - (np.maximum(((-2.0)), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) - (data["flux_std"]))) +
                0.100000*np.tanh(((((((((np.where(((data["distmod"]) / 2.0) > -1, -1.0, 0.367879 )) / 2.0)) * 2.0)) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((((-2.0) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((((((((((data["flux_d0_pb1"]) - (np.tanh(((7.0)))))) < (data["flux_w_mean"]))*1.)) + (data["flux_std"]))) - (data["flux_d0_pb1"]))) +
                0.100000*np.tanh(((data["0__skewness_x"]) + ((((((data["3__skewness_x"]) + ((((data["3__kurtosis_x"]) + (data["detected_flux_skew"]))/2.0)))/2.0)) + (-2.0))))) +
                0.100000*np.tanh(((((np.where(0.367879 > -1, ((data["ddf"]) + (((((data["flux_err_std"]) * 2.0)) * 2.0))), data["flux_err_std"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_skew"]))))) +
                0.100000*np.tanh(((((data["flux_max"]) + (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh((((data["1__kurtosis_y"]) + (np.minimum(((((data["flux_ratio_sq_sum"]) + (np.where(data["2__kurtosis_y"] > -1, data["2__kurtosis_y"], data["3__kurtosis_x"] ))))), ((data["1__fft_coefficient__coeff_0__attr__abs__y"])))))/2.0)) +
                0.100000*np.tanh(((data["flux_ratio_sq_sum"]) * (np.tanh((data["detected_flux_ratio_sq_skew"]))))) +
                0.100000*np.tanh(((((((np.where(data["flux_err_std"] > -1, ((data["flux_err_std"]) * 2.0), data["flux_err_std"] )) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["detected_flux_skew"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.maximum(((-3.0)), ((data["3__kurtosis_x"]))), data["hostgal_photoz"] )) +
                0.100000*np.tanh(np.maximum(((data["0__skewness_y"])), ((data["1__kurtosis_x"])))) +
                0.100000*np.tanh(((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_ratio_sq_skew"]) / 2.0)) - (data["detected_flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh((((((data["2__kurtosis_x"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (((data["2__skewness_x"]) - (((data["detected_flux_by_flux_ratio_sq_skew"]) - (data["2__skewness_x"]))))))/2.0)) +
                0.100000*np.tanh(((((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0)) - (data["flux_d0_pb1"]))) * 2.0)) + (data["flux_skew"]))) +
                0.100000*np.tanh(((((data["flux_err_std"]) * 2.0)) * 2.0)) +
                0.100000*np.tanh((((data["flux_d1_pb5"]) > ((((data["flux_ratio_sq_sum"]) + (data["detected_mjd_diff"]))/2.0)))*1.)) +
                0.100000*np.tanh((((data["flux_err_std"]) + (data["flux_d1_pb5"]))/2.0)) +
                0.100000*np.tanh(((-2.0) + (np.maximum(((np.maximum(((-2.0)), ((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))))))), ((data["detected_mjd_diff"])))))) +
                0.100000*np.tanh(((np.where(data["2__skewness_x"]<0, data["flux_err_std"], ((((((data["2__skewness_x"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["detected_flux_w_mean"]))/2.0) )) * 2.0)) +
                0.100000*np.tanh((((((((data["0__skewness_x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) + (data["0__skewness_x"]))/2.0)) +
                0.100000*np.tanh((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) < (data["detected_flux_min"]))*1.)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_ratio_sq_sum"] )) / 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_sum"] > -1, data["1__skewness_x"], data["3__kurtosis_x"] )) +
                0.100000*np.tanh((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["flux_skew"]) + (data["flux_ratio_sq_skew"]))))/2.0)))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]>0, data["5__kurtosis_x"], np.where(np.minimum(((data["1__skewness_y"])), ((data["flux_err_std"])))>0, np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["5__kurtosis_x"]))), data["flux_err_std"] ) )) +
                0.100000*np.tanh(np.minimum(((np.minimum((((((data["detected_mjd_diff"]) + (np.minimum(((data["flux_mean"])), ((data["0__fft_coefficient__coeff_1__attr__abs__y"])))))/2.0))), ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))))), ((data["3__kurtosis_x"])))) +
                0.100000*np.tanh((((np.minimum(((((data["flux_ratio_sq_skew"]) / 2.0))), ((data["1__skewness_y"])))) + (data["1__skewness_y"]))/2.0)) +
                0.100000*np.tanh(np.maximum(((data["flux_dif3"])), ((data["detected_flux_ratio_sq_sum"])))) +
                0.100000*np.tanh(np.where(data["flux_max"] > -1, -3.0, data["2__skewness_x"] )) +
                0.100000*np.tanh(((((np.where(((-2.0) - (data["distmod"]))<0, -2.0, data["ddf"] )) - (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, data["0__skewness_y"], 3.0 )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb3"]>0, ((data["4__kurtosis_x"]) * 2.0), ((np.where(data["flux_d1_pb3"]>0, ((data["detected_flux_median"]) + (data["ddf"])), data["flux_d1_pb3"] )) / 2.0) )))

    def GP_class_67(self,data):
        return (-1.801807 +
                0.100000*np.tanh(((data["4__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (np.minimum(((data["5__kurtosis_x"])), ((data["4__kurtosis_x"])))))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((data["3__skewness_x"]) + (((data["5__kurtosis_x"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((((((data["4__kurtosis_x"]) + (data["detected_flux_min"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["4__skewness_x"]))) + (((data["detected_flux_min"]) + (data["3__kurtosis_x"]))))) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"]))) + (((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["4__kurtosis_x"]) + (data["distmod"]))))))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["4__kurtosis_x"]))) + (((((data["3__kurtosis_x"]) + (data["4__kurtosis_x"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(((((((((data["3__kurtosis_x"]) + (data["distmod"]))) + (data["3__kurtosis_x"]))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) + ((((((data["flux_by_flux_ratio_sq_skew"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))/2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((((data["5__kurtosis_x"]) - (0.0))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((data["distmod"])))) + (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(np.minimum(((data["3__kurtosis_x"])), ((data["hostgal_photoz_err"])))) +
                0.100000*np.tanh(((((((data["detected_flux_w_mean"]) + (data["hostgal_photoz_err"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((((data["hostgal_photoz_err"]) + (data["flux_dif2"]))) + (data["4__kurtosis_x"]))/2.0)) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((np.tanh((((data["flux_ratio_sq_skew"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["hostgal_photoz_err"])), ((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.minimum(((data["flux_dif2"])), ((data["detected_flux_min"]))))), ((data["2__kurtosis_x"]))))))))))), ((data["flux_ratio_sq_skew"])))) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) - (data["0__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((((data["2__kurtosis_y"]) + (((data["2__skewness_x"]) + (np.minimum(((data["ddf"])), ((data["4__kurtosis_x"]))))))))), ((np.minimum(((data["distmod"])), ((data["distmod"]))))))) +
                0.100000*np.tanh((((((data["distmod"]) + (((((data["distmod"]) * 2.0)) + (((data["mjd_size"]) - (data["detected_mjd_size"]))))))/2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["distmod"])), ((np.minimum(((data["distmod"])), ((data["distmod"])))))))), ((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["5__kurtosis_x"])))) +
                0.100000*np.tanh(((((((((data["5__kurtosis_x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_d1_pb0"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_dif2"])), ((data["flux_dif2"])))) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["flux_by_flux_ratio_sq_skew"]) + (data["2__kurtosis_x"]))) + (data["flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(((((((data["flux_dif2"]) + (data["3__skewness_y"]))/2.0)) + (np.minimum(((data["0__kurtosis_y"])), ((data["3__skewness_y"])))))/2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((np.minimum(((((data["2__kurtosis_y"]) + (data["flux_d1_pb5"])))), ((data["2__kurtosis_y"]))))))) / 2.0)) +
                0.100000*np.tanh(np.minimum(((((((data["flux_by_flux_ratio_sq_skew"]) - (data["detected_mjd_diff"]))) * 2.0))), ((data["4__skewness_x"])))) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) - (((np.where(data["detected_mjd_diff"] > -1, ((((data["detected_mjd_diff"]) - (data["3__skewness_x"]))) * 2.0), data["3__skewness_x"] )) * 2.0)))) +
                0.100000*np.tanh((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["distmod"]))/2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((np.tanh((data["4__skewness_x"]))) - (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__skewness_y"]))))) +
                0.100000*np.tanh(np.minimum(((data["4__kurtosis_x"])), ((data["3__kurtosis_x"])))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif3"] )) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.minimum(((data["hostgal_photoz_err"])), ((data["1__skewness_y"])))) + (data["detected_flux_std"]))) + (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["5__kurtosis_x"]) - (data["detected_mjd_diff"]))) - ((((data["0__kurtosis_x"]) + (data["detected_mjd_diff"]))/2.0)))) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d1_pb5"])), ((np.minimum(((data["mjd_size"])), ((data["2__kurtosis_x"]))))))) + (((data["2__kurtosis_y"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((data["detected_flux_std"]) * ((((14.21325874328613281)) * (((data["flux_dif2"]) * ((7.0)))))))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb5"] > -1, ((data["detected_flux_dif3"]) - (data["0__skewness_x"])), ((np.where(data["3__kurtosis_x"]>0, data["flux_d1_pb5"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["flux_d1_pb5"])) )) +
                0.100000*np.tanh(((data["mjd_size"]) + (data["flux_d0_pb5"]))) +
                0.100000*np.tanh(((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) + (data["detected_flux_diff"]))) +
                0.100000*np.tanh((((((data["ddf"]) + (data["distmod"]))/2.0)) + (((data["2__kurtosis_y"]) + (((data["detected_flux_dif3"]) + (data["4__kurtosis_y"]))))))) +
                0.100000*np.tanh(((((data["flux_d0_pb1"]) + ((((7.57827568054199219)) - (data["detected_flux_std"]))))) * (((((((data["detected_flux_std"]) - (data["flux_d0_pb1"]))) * 2.0)) * 2.0)))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["flux_dif2"]))) + (data["flux_dif2"]))) +
                0.100000*np.tanh(np.minimum(((-1.0)), ((np.where(data["3__skewness_y"]>0, data["3__skewness_y"], data["3__skewness_y"] ))))) +
                0.100000*np.tanh(((np.maximum(((data["flux_dif2"])), (((14.82015228271484375))))) * ((((((14.82015228271484375)) * ((14.82015228271484375)))) * (data["flux_dif2"]))))) +
                0.100000*np.tanh(((((data["distmod"]) + (data["detected_flux_std"]))) * ((((8.74618148803710938)) + (data["detected_flux_std"]))))) +
                0.100000*np.tanh(np.minimum(((data["3__skewness_y"])), ((((((data["flux_d0_pb5"]) + (data["flux_d0_pb5"]))) * 2.0))))) +
                0.100000*np.tanh(((((((((data["flux_max"]) - (data["detected_mjd_diff"]))) - (data["detected_flux_err_median"]))) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_std"]) * (((data["detected_flux_std"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["2__skewness_y"]))) +
                0.100000*np.tanh(np.where(((data["detected_flux_dif2"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))>0, ((np.where(data["detected_mean"]>0, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_std"] )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["1__fft_coefficient__coeff_0__attr__abs__x"] )) +
                0.100000*np.tanh(np.maximum(((((data["detected_flux_dif2"]) + (data["ddf"])))), ((data["detected_flux_std"])))) +
                0.100000*np.tanh(((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (((data["detected_mjd_diff"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["flux_dif2"]<0, np.where(data["detected_flux_std"]<0, data["detected_flux_err_skew"], data["flux_max"] ), data["detected_flux_std"] )<0, data["0__kurtosis_x"], data["detected_flux_dif2"] )) +
                0.100000*np.tanh(((((((data["flux_d1_pb5"]) * (data["flux_d0_pb3"]))) - (np.where(data["0__skewness_x"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d1_pb0"] )))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_min"]<0, data["5__fft_coefficient__coeff_0__attr__abs__y"], np.minimum(((data["distmod"])), ((data["5__fft_coefficient__coeff_0__attr__abs__y"]))) )) - (data["flux_d1_pb1"]))) +
                0.100000*np.tanh(np.where((((data["flux_max"]) > (data["1__fft_coefficient__coeff_0__attr__abs__x"]))*1.)>0, ((data["distmod"]) + (data["detected_flux_std"])), -2.0 )) +
                0.100000*np.tanh(((((data["detected_flux_dif2"]) - (np.where(data["detected_flux_dif2"] > -1, data["detected_mjd_diff"], (((data["detected_flux_dif2"]) > (data["detected_mjd_diff"]))*1.) )))) - (data["detected_mean"]))) +
                0.100000*np.tanh(((((((data["5__kurtosis_y"]) - (data["flux_by_flux_ratio_sq_sum"]))) + (((((data["1__skewness_y"]) - (data["detected_flux_err_min"]))) + (data["5__kurtosis_y"]))))) - (data["detected_flux_err_median"]))) +
                0.100000*np.tanh(((((((((data["flux_max"]) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["5__skewness_x"]) * (data["2__fft_coefficient__coeff_0__attr__abs__y"])), data["2__skewness_x"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) - (data["flux_d1_pb1"]))<0, -2.0, data["flux_dif2"] )) +
                0.100000*np.tanh(np.where(((data["detected_flux_std"]) - (data["flux_d0_pb2"]))>0, np.where(data["flux_err_std"]>0, data["detected_flux_std"], data["1__kurtosis_x"] ), data["flux_err_std"] )) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.where(data["hostgal_photoz"] > -1, np.tanh((data["detected_flux_std"])), data["hostgal_photoz"] ), ((data["hostgal_photoz_err"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(np.where(data["flux_dif2"]<0, data["flux_dif2"], data["flux_err_skew"] )<0, np.where(data["flux_err_skew"]<0, data["flux_err_skew"], data["flux_dif2"] ), data["ddf"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb2"]<0, np.where(data["flux_d0_pb4"]<0, data["flux_err_skew"], (((data["flux_d0_pb1"]) < (data["flux_std"]))*1.) ), ((data["detected_flux_std"]) * 2.0) )) +
                0.100000*np.tanh(((((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.maximum(((-3.0)), ((((data["0__kurtosis_y"]) * 2.0))))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((((np.where(data["detected_flux_dif2"]<0, data["0__skewness_x"], ((((data["detected_flux_std"]) * (data["3__skewness_y"]))) - (data["0__skewness_x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((((data["0__kurtosis_x"]) + (((np.minimum(((data["flux_err_max"])), ((((data["5__kurtosis_y"]) / 2.0))))) + (data["5__kurtosis_y"])))))))) +
                0.100000*np.tanh(((np.where(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_d1_pb5"], data["2__fft_coefficient__coeff_1__attr__abs__x"] ) > -1, data["flux_d1_pb5"], data["2__skewness_y"] )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((np.where(data["flux_d0_pb4"]<0, data["4__fft_coefficient__coeff_0__attr__abs__x"], ((data["detected_flux_std"]) + (np.minimum(((data["distmod"])), ((data["detected_flux_std"]))))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"] > -1, (((((np.tanh((data["detected_flux_dif2"]))) > (data["detected_mjd_diff"]))*1.)) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["detected_mjd_diff"]>0, data["flux_err_skew"], data["4__fft_coefficient__coeff_1__attr__abs__x"] ), data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((np.where(data["2__skewness_y"]>0, data["3__kurtosis_x"], np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_diff"]))) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((((((data["detected_flux_std"]) > (data["flux_d0_pb1"]))*1.)) > (data["flux_d0_pb1"]))*1.), data["flux_d0_pb1"] )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["2__skewness_x"]))) + (data["0__kurtosis_x"]))))) * (data["detected_flux_diff"]))) +
                0.100000*np.tanh(((np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["4__fft_coefficient__coeff_1__attr__abs__x"], ((data["detected_flux_std"]) * 2.0) )) - (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["hostgal_photoz_err"]<0, data["2__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["3__fft_coefficient__coeff_1__attr__abs__y"]>0, data["4__skewness_x"], data["flux_min"] ) ) )) +
                0.100000*np.tanh(np.where(np.where((((data["distmod"]) > (data["1__fft_coefficient__coeff_1__attr__abs__y"]))*1.)>0, data["flux_dif2"], data["1__fft_coefficient__coeff_1__attr__abs__y"] )>0, (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) > (data["1__fft_coefficient__coeff_1__attr__abs__y"]))*1.), -3.0 )) +
                0.100000*np.tanh(((((((((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) - (data["detected_mjd_diff"]))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_std"])), ((((data["flux_d0_pb1"]) * (np.minimum(((data["flux_max"])), ((np.tanh((data["detected_flux_diff"]))))))))))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"] > -1, (((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_max"]))*1.)) * 2.0)) - (data["1__fft_coefficient__coeff_0__attr__abs__y"])), data["1__fft_coefficient__coeff_0__attr__abs__y"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb0"]<0, data["0__fft_coefficient__coeff_0__attr__abs__y"], ((((data["1__skewness_y"]) * (data["detected_flux_w_mean"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"])) )) +
                0.100000*np.tanh(np.where(np.where(data["hostgal_photoz_err"]<0, np.where(data["detected_mean"] > -1, data["0__skewness_x"], data["hostgal_photoz_err"] ), data["hostgal_photoz_err"] )<0, data["hostgal_photoz_err"], data["2__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((data["detected_flux_dif2"]) + (((((data["4__skewness_y"]) * (data["4__skewness_y"]))) + (((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) * (data["4__skewness_y"]))) + (data["distmod"]))))))) +
                0.100000*np.tanh(((((((((data["detected_flux_diff"]) * (data["1__skewness_y"]))) - (data["flux_d1_pb0"]))) - (data["detected_mjd_diff"]))) - (((data["flux_d1_pb0"]) - (data["detected_flux_mean"]))))) +
                0.100000*np.tanh(((((np.where(data["flux_d0_pb0"]>0, ((data["detected_flux_std"]) - (data["flux_d0_pb0"])), ((data["flux_err_skew"]) - (data["1__skewness_x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"] > -1, ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) * (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["detected_mjd_diff"]<0, data["flux_d0_pb4"], np.where(data["detected_mjd_diff"]>0, np.where(data["flux_d0_pb0"]<0, data["flux_err_min"], data["3__fft_coefficient__coeff_0__attr__abs__x"] ), data["detected_mjd_diff"] ) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]>0, data["4__kurtosis_y"], ((data["flux_d1_pb3"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (np.where(data["5__skewness_x"]>0, data["flux_d1_pb3"], data["4__skewness_x"] ))))) )) +
                0.100000*np.tanh(((((np.where(np.where(data["detected_flux_diff"]<0, data["flux_max"], data["flux_diff"] )<0, data["flux_diff"], data["flux_max"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"]<0, data["0__fft_coefficient__coeff_1__attr__abs__y"], np.where(data["3__kurtosis_x"]<0, data["detected_flux_err_mean"], ((data["distmod"]) - (data["detected_flux_err_median"])) ) )) +
                0.100000*np.tanh((((((data["detected_mjd_diff"]) < (np.tanh((data["detected_flux_dif2"]))))*1.)) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_d0_pb5"], ((data["3__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_ratio_sq_skew"])) )) +
                0.100000*np.tanh(np.where(np.maximum(((data["2__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) > -1, np.where(data["mjd_diff"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__x"], data["detected_mjd_size"] ), data["2__skewness_y"] )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, (-1.0*((data["4__skewness_y"]))), (((data["3__skewness_x"]) + (data["flux_d1_pb1"]))/2.0) )) +
                0.100000*np.tanh(((data["flux_err_max"]) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__skewness_y"]))))) +
                0.100000*np.tanh(((np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_err_skew"], ((((np.where(data["flux_std"]<0, data["hostgal_photoz_err"], data["flux_err_max"] )) * 2.0)) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (((data["detected_flux_std"]) + (data["distmod"]))))<0, data["distmod"], ((data["detected_flux_std"]) - (data["distmod"])) )) +
                0.100000*np.tanh(np.where((((data["detected_mjd_diff"]) < (np.tanh(((((data["detected_flux_dif2"]) + (data["detected_mjd_diff"]))/2.0)))))*1.)>0, data["detected_flux_dif2"], -3.0 )) +
                0.100000*np.tanh(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, ((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_d1_pb5"])))) )) +
                0.100000*np.tanh(((((((((data["5__kurtosis_y"]) + (((data["flux_d1_pb5"]) + (data["distmod"]))))) + (data["flux_median"]))) + (data["1__kurtosis_y"]))) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["5__skewness_y"]<0, ((((data["5__kurtosis_y"]) + (data["distmod"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["1__skewness_y"] )) +
                0.100000*np.tanh(np.where(data["detected_mjd_diff"]<0, data["mjd_diff"], ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) +
                0.100000*np.tanh(np.where((((data["detected_flux_w_mean"]) < (np.where(data["flux_mean"]<0, data["flux_d1_pb4"], data["flux_mean"] )))*1.)>0, (-1.0*((data["flux_d1_pb4"]))), data["flux_d1_pb4"] )) +
                0.100000*np.tanh(((((data["flux_skew"]) - (data["3__kurtosis_y"]))) - (np.maximum(((data["flux_d1_pb1"])), ((data["3__kurtosis_y"])))))) +
                0.100000*np.tanh(((data["4__skewness_y"]) * (((data["3__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["flux_d0_pb4"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))))) +
                0.100000*np.tanh(((np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["2__kurtosis_x"])), data["3__kurtosis_x"] ), data["3__kurtosis_x"] )) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_std"]>0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__kurtosis_x"], data["flux_std"] ), data["detected_flux_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_std"]<0, ((((data["hostgal_photoz_err"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) - (data["detected_flux_by_flux_ratio_sq_skew"])), np.where(data["detected_flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_by_flux_ratio_sq_skew"], data["4__skewness_y"] ) )) +
                0.100000*np.tanh(((((((((((data["flux_max"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_skew"] > -1, ((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_ratio_sq_skew"]))) * 2.0), 2.0 )) * 2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["distmod"]) - (data["0__fft_coefficient__coeff_0__attr__abs__x"])), np.tanh((data["3__fft_coefficient__coeff_1__attr__abs__x"])) ), data["detected_flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((data["detected_flux_dif2"]) - (data["detected_flux_err_median"]))) * 2.0)) + (data["4__kurtosis_y"]))) +
                0.100000*np.tanh(np.where(np.where(((data["flux_err_std"]) + ((((data["1__skewness_y"]) + (data["flux_err_std"]))/2.0)))>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )>0, data["flux_err_std"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(((data["distmod"]) * 2.0) > -1, (((((-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)) * 2.0), ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) * (np.where(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * (data["detected_flux_err_skew"]))) + (data["detected_flux_err_skew"]))>0, data["detected_flux_err_skew"], data["2__fft_coefficient__coeff_1__attr__abs__y"] )))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, data["detected_flux_diff"], np.where(data["flux_median"]>0, data["3__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["flux_median"] > -1, data["flux_median"], data["flux_median"] ) ) )))

    def GP_class_88(self,data):
        return (-1.503109 +
                0.100000*np.tanh(((((data["distmod"]) + (((((data["distmod"]) * 2.0)) * 2.0)))) + (np.where(data["distmod"] > -1, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["distmod"] )))) +
                0.100000*np.tanh(((np.where(data["4__kurtosis_x"]>0, data["distmod"], data["distmod"] )) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["distmod"]) + (((data["distmod"]) - (data["3__kurtosis_x"]))))))) +
                0.100000*np.tanh((((((np.minimum(((data["distmod"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (((data["distmod"]) + (((data["distmod"]) - (data["2__kurtosis_x"]))))))/2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["distmod"]) + ((((((data["distmod"]) > (data["distmod"]))*1.)) - (data["4__kurtosis_x"]))))) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))) + (data["distmod"]))) + (((data["distmod"]) * 2.0)))) +
                0.100000*np.tanh(((np.where(((data["distmod"]) - (data["distmod"])) > -1, np.minimum(((data["distmod"])), ((data["distmod"]))), ((data["0__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0) )) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((np.where(data["2__skewness_x"]>0, -3.0, ((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"])) )) * 2.0)))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"])))) * 2.0))), ((((data["distmod"]) - (data["2__kurtosis_x"])))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_mean"]) - (data["detected_flux_min"]))) +
                0.100000*np.tanh(((((np.where(data["distmod"] > -1, data["detected_mjd_diff"], -1.0 )) + (-1.0))) * 2.0)) +
                0.100000*np.tanh(((((((((((data["flux_skew"]) - (data["flux_skew"]))) - (data["flux_skew"]))) + (data["detected_mjd_diff"]))) - (((data["flux_skew"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh((((data["detected_mjd_diff"]) + (((((((data["distmod"]) + ((-1.0*((data["4__skewness_x"])))))) * 2.0)) * 2.0)))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (((((data["distmod"]) - (data["2__skewness_x"]))) - (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["4__kurtosis_x"]))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, ((data["3__kurtosis_x"]) - (((((((data["3__skewness_x"]) * 2.0)) * 2.0)) * 2.0))), data["hostgal_photoz"] )) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((((data["distmod"]) * 2.0)) + (data["distmod"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((((((data["distmod"]) - (data["flux_skew"]))) - (data["1__kurtosis_x"]))) - (data["flux_skew"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["2__skewness_x"]<0, np.where(data["2__skewness_x"] > -1, data["distmod"], ((data["distmod"]) + (data["distmod"])) ), -3.0 )) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]<0, np.where(data["distmod"]<0, np.where(-3.0<0, data["distmod"], data["3__skewness_x"] ), data["detected_mjd_diff"] ), -2.0 )) +
                0.100000*np.tanh(((np.where(np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"]))) > -1, data["distmod"], data["distmod"] )) * 2.0)) +
                0.100000*np.tanh(((((((np.where(data["hostgal_photoz"] > -1, data["detected_mjd_diff"], data["flux_skew"] )) - (data["detected_flux_median"]))) - (data["3__skewness_x"]))) - (data["flux_skew"]))) +
                0.100000*np.tanh((((((((-1.0*((data["flux_skew"])))) * 2.0)) - (np.where(data["flux_ratio_sq_sum"] > -1, data["3__skewness_x"], ((data["flux_d1_pb2"]) * (data["detected_mjd_size"])) )))) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(((((np.where(data["1__kurtosis_x"]>0, -2.0, np.where(data["1__kurtosis_x"]<0, ((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0), data["1__fft_coefficient__coeff_1__attr__abs__y"] ) )) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + (((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((data["distmod"])), ((data["distmod"]))))))) * 2.0)))) +
                0.100000*np.tanh(np.where(data["flux_skew"]>0, data["hostgal_photoz"], ((data["distmod"]) * 2.0) )) +
                0.100000*np.tanh(((((((data["4__skewness_x"]) * (data["detected_mean"]))) / 2.0)) - ((((12.28340435028076172)) * (data["4__skewness_x"]))))) +
                0.100000*np.tanh(((((((data["detected_mjd_diff"]) - (data["detected_flux_diff"]))) - (np.maximum(((data["detected_flux_std"])), ((data["1__skewness_x"])))))) - (data["flux_skew"]))) +
                0.100000*np.tanh(((((np.minimum(((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"])))) * 2.0))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0)) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["detected_mean"]) - (data["4__skewness_x"]))) * 2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (((data["detected_mjd_diff"]) - (0.367879))))*1.)))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) * 2.0))), ((data["distmod"])))) + (((((data["distmod"]) + (data["detected_mjd_diff"]))) + (data["distmod"]))))) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]>0, np.where(data["flux_skew"]>0, -3.0, data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(np.minimum(((((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) * 2.0)) * 2.0)) * 2.0))), ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, ((data["detected_mjd_diff"]) - (data["distmod"])), ((data["distmod"]) + (data["distmod"])) )) +
                0.100000*np.tanh(((((data["distmod"]) + (data["distmod"]))) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.minimum(((((data["distmod"]) - (data["1__skewness_x"])))), ((np.where(data["detected_mjd_diff"]>0, data["detected_mjd_diff"], data["flux_skew"] ))))) - (data["flux_d1_pb2"]))) * 2.0)) +
                0.100000*np.tanh(((((((((-1.0) + (data["detected_mjd_diff"]))) + (data["distmod"]))) * 2.0)) + (((-1.0) * 2.0)))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((((((data["flux_err_min"]) + (data["distmod"]))) * 2.0)) + (data["detected_mjd_diff"]))) + (data["distmod"]))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((np.tanh((((np.tanh((((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)))) * 2.0)))) * 2.0))))) +
                0.100000*np.tanh(((((data["detected_mean"]) + (data["distmod"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) - (data["detected_flux_by_flux_ratio_sq_sum"]))) - (data["flux_skew"]))) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))))*1.)))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (((data["distmod"]) * 2.0)))) + (((data["distmod"]) + (-3.0))))) +
                0.100000*np.tanh(((((((((data["detected_mean"]) - (data["flux_d0_pb3"]))) - (data["1__kurtosis_x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) + ((((((((data["distmod"]) + (data["distmod"]))) + (np.minimum(((data["hostgal_photoz"])), ((data["distmod"])))))/2.0)) * 2.0)))) +
                0.100000*np.tanh(((((((((-1.0) + (data["detected_mjd_diff"]))) * 2.0)) * 2.0)) + (-1.0))) +
                0.100000*np.tanh(np.where(np.minimum(((data["detected_mjd_diff"])), ((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_ratio_sq_skew"])))), ((data["detected_flux_ratio_sq_skew"]))))))>0, data["detected_mjd_diff"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - ((((np.where(data["flux_skew"] > -1, data["detected_mjd_diff"], data["flux_err_min"] )) > (((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (((((data["detected_mjd_diff"]) + (((data["flux_err_min"]) + (data["detected_mjd_diff"]))))) + (data["flux_err_min"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_skew"]))) +
                0.100000*np.tanh(((data["distmod"]) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(((((((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((data["3__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["detected_mjd_diff"])))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["distmod"]) + (((data["0__skewness_x"]) - ((((data["0__skewness_x"]) > (data["detected_mjd_diff"]))*1.)))))) + (data["distmod"]))) +
                0.100000*np.tanh(((np.where(data["flux_skew"]>0, -2.0, ((((data["flux_err_min"]) * 2.0)) * 2.0) )) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, data["detected_mjd_diff"], ((data["distmod"]) + (np.where(data["distmod"] > -1, data["distmod"], np.minimum(((data["flux_by_flux_ratio_sq_sum"])), ((data["flux_d0_pb5"]))) ))) )) +
                0.100000*np.tanh(((((np.where(((data["flux_err_median"]) * 2.0) > -1, ((data["detected_mean"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"])), np.minimum(((data["detected_mean"])), ((data["detected_mean"]))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(((((data["distmod"]) + ((((data["detected_mjd_diff"]) + (((((data["flux_d0_pb0"]) + (data["detected_mjd_diff"]))) + (data["distmod"]))))/2.0)))) + (data["distmod"]))) +
                0.100000*np.tanh(np.where((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_by_flux_ratio_sq_sum"]))/2.0) > -1, ((data["detected_mjd_diff"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_mjd_diff"])) )) +
                0.100000*np.tanh(((((data["detected_flux_dif3"]) + (((((((data["flux_err_min"]) + (data["detected_mean"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)))) + (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["flux_err_min"], (-1.0*((data["1__fft_coefficient__coeff_1__attr__abs__y"]))) )) + (data["detected_flux_ratio_sq_skew"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"] > -1, ((((((((data["flux_median"]) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["flux_d0_pb0"]))) * 2.0), data["flux_d0_pb2"] )) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((((((data["distmod"]) + (data["distmod"]))) + (data["detected_flux_ratio_sq_skew"]))) + (data["distmod"])))))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.where(data["flux_median"]<0, data["0__fft_coefficient__coeff_1__attr__abs__x"], ((data["detected_mjd_diff"]) + (np.tanh((((data["flux_err_min"]) * 2.0))))) )) +
                0.100000*np.tanh(np.where(np.where(data["detected_flux_err_skew"]<0, data["detected_flux_diff"], data["detected_mjd_diff"] ) > -1, data["detected_mjd_diff"], data["detected_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.minimum(((((((np.minimum(((data["detected_mjd_diff"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) * 2.0)) * 2.0))), ((((data["detected_mean"]) * 2.0))))) +
                0.100000*np.tanh(np.minimum(((((((((data["flux_median"]) - (data["detected_flux_max"]))) - (data["flux_std"]))) - (data["1__kurtosis_x"])))), ((data["flux_median"])))) +
                0.100000*np.tanh(((data["flux_d0_pb5"]) + (((np.where(data["detected_mjd_diff"] > -1, ((-2.0) + (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) * 2.0)))) +
                0.100000*np.tanh(((((data["flux_err_min"]) + (((data["hostgal_photoz"]) + (((data["flux_err_min"]) + (data["distmod"]))))))) * 2.0)) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["distmod"]))) +
                0.100000*np.tanh((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((np.where(data["detected_mean"]<0, ((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"])), data["0__fft_coefficient__coeff_0__attr__abs__x"] )) * 2.0)))/2.0)) +
                0.100000*np.tanh(((np.where(data["flux_err_max"] > -1, data["0__fft_coefficient__coeff_1__attr__abs__x"], data["distmod"] )) + (np.minimum(((data["hostgal_photoz"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_err_min"])), ((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (data["distmod"]))))))), ((data["distmod"])))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (((data["detected_mjd_diff"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (data["flux_err_min"]))) +
                0.100000*np.tanh(((((np.where(data["0__skewness_y"]>0, data["detected_flux_ratio_sq_skew"], np.where(data["flux_err_min"] > -1, data["flux_err_min"], data["detected_flux_ratio_sq_skew"] ) )) * 2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["flux_dif3"]))) + (((data["4__skewness_y"]) + (((((data["detected_mjd_diff"]) - (data["detected_flux_mean"]))) / 2.0)))))) +
                0.100000*np.tanh(np.where(np.maximum(((data["flux_d1_pb1"])), ((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (data["detected_flux_diff"]))))) > -1, data["detected_flux_ratio_sq_skew"], data["5__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_d0_pb1"]<0, data["2__fft_coefficient__coeff_1__attr__abs__x"], np.where((((-1.0*((data["0__fft_coefficient__coeff_0__attr__abs__x"])))) / 2.0) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_d0_pb1"] ) )) +
                0.100000*np.tanh((((data["distmod"]) < (np.minimum(((data["distmod"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))))*1.)) +
                0.100000*np.tanh((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((np.minimum(((data["flux_err_min"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (np.minimum(((data["4__skewness_y"])), ((data["detected_mean"])))))))/2.0)) +
                0.100000*np.tanh(((((data["flux_err_min"]) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) + ((((data["distmod"]) + ((((((data["distmod"]) + (data["distmod"]))) + (data["flux_d1_pb4"]))/2.0)))/2.0)))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (data["2__skewness_y"]))) +
                0.100000*np.tanh(((((((data["detected_mean"]) + (data["distmod"]))) * 2.0)) - (np.where(data["mjd_diff"] > -1, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_1__attr__abs__y"] )))) +
                0.100000*np.tanh(np.where(data["flux_d1_pb4"] > -1, data["flux_err_min"], ((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) )) +
                0.100000*np.tanh(((((data["flux_err_min"]) * 2.0)) / 2.0)) +
                0.100000*np.tanh((((data["distmod"]) + (data["flux_d0_pb0"]))/2.0)) +
                0.100000*np.tanh(((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["detected_flux_err_mean"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["distmod"]))/2.0)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["detected_flux_diff"]))/2.0)) + (np.maximum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["flux_err_min"]) * 2.0))))))) +
                0.100000*np.tanh(np.tanh((((data["flux_min"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((data["3__skewness_y"])), ((data["3__skewness_y"])))) + (0.367879)))), ((np.minimum(((data["flux_err_min"])), ((((data["5__skewness_y"]) + (data["flux_err_min"]))))))))) +
                0.100000*np.tanh(((data["detected_mjd_diff"]) - (np.where(data["flux_d0_pb3"]<0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["3__fft_coefficient__coeff_0__attr__abs__x"])), ((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["4__skewness_x"])) )))) +
                0.100000*np.tanh(((data["5__kurtosis_y"]) + (np.minimum(((data["detected_flux_ratio_sq_skew"])), ((((np.tanh((data["4__kurtosis_y"]))) + (data["detected_mjd_diff"])))))))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__kurtosis_y"]))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) +
                0.100000*np.tanh(np.where(data["flux_err_min"] > -1, data["detected_flux_ratio_sq_skew"], ((data["detected_flux_ratio_sq_skew"]) * 2.0) )) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_d0_pb0"]))) + ((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_median"]))) < (np.tanh((data["0__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))) +
                0.100000*np.tanh(np.where(((data["distmod"]) + (data["1__skewness_y"]))>0, (((data["detected_flux_ratio_sq_skew"]) > (np.minimum(((data["distmod"])), ((data["5__skewness_y"])))))*1.), data["distmod"] )) +
                0.100000*np.tanh(((((data["3__skewness_y"]) + (data["4__kurtosis_y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["hostgal_photoz"] > -1, ((data["flux_err_min"]) - (data["detected_flux_min"])), data["hostgal_photoz"] )) +
                0.100000*np.tanh((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_median"], data["0__fft_coefficient__coeff_1__attr__abs__y"] )) + (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) / 2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["mjd_size"]<0, data["flux_median"], np.where(data["detected_flux_min"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], ((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d1_pb5"]))) + (data["5__kurtosis_x"])) ) )) +
                0.100000*np.tanh(np.maximum(((data["detected_flux_ratio_sq_skew"])), ((data["0__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(((np.where(data["flux_err_std"] > -1, data["flux_median"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["distmod"]) + (np.where(np.tanh((data["detected_flux_min"]))>0, data["detected_flux_err_skew"], data["4__skewness_y"] )))) +
                0.100000*np.tanh(((data["distmod"]) + (((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(3.0 > -1, data["distmod"], ((data["detected_mean"]) - ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.))) )) +
                0.100000*np.tanh((((((((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (data["3__skewness_y"]))/2.0)) - (data["1__kurtosis_x"]))) - ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)))) - (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(np.where(data["flux_err_min"]>0, data["flux_err_min"], data["flux_err_min"] )>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["flux_err_min"], data["flux_err_min"] ) )) +
                0.100000*np.tanh(np.minimum(((((data["4__skewness_y"]) + (data["3__skewness_y"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_median"]))) + (data["flux_median"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["detected_mean"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__kurtosis_x"]>0, data["detected_mjd_diff"], np.tanh((data["flux_err_min"])) ) ))), ((0.0)))) +
                0.100000*np.tanh(((((((((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))) - (((data["detected_flux_max"]) * 2.0)))) - (data["detected_flux_min"]))) - (data["flux_d0_pb3"]))))

    def GP_class_90(self,data):
        return (-0.436273 +
                0.100000*np.tanh(np.minimum(((-3.0)), ((np.minimum(((np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((3.141593))))), ((-3.0))))))) +
                0.100000*np.tanh(((np.minimum(((-3.0)), ((-2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((-3.0)), ((-3.0)))) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((np.minimum(((np.minimum(((-3.0)), ((data["3__kurtosis_y"]))))), ((-3.0))))), ((-3.0))))))) +
                0.100000*np.tanh((((-3.0) + (np.minimum(((np.minimum(((-3.0)), ((np.minimum(((-3.0)), ((data["4__kurtosis_x"])))))))), ((np.minimum(((data["2__skewness_x"])), ((-3.0))))))))/2.0)) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((data["distmod"])), ((data["distmod"])))>0, data["flux_by_flux_ratio_sq_skew"], -2.0 ))), ((data["distmod"])))) +
                0.100000*np.tanh(((((np.minimum(((np.minimum(((((((data["distmod"]) * 2.0)) * 2.0))), ((data["3__kurtosis_x"]))))), ((data["distmod"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["flux_ratio_sq_skew"]) + (data["distmod"])))), ((((np.minimum(((data["distmod"])), ((data["distmod"])))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((np.minimum(((np.minimum(((data["4__kurtosis_x"])), ((data["distmod"]))))), ((data["distmod"])))) + (data["2__skewness_x"])))), ((np.minimum(((data["distmod"])), ((data["4__kurtosis_x"]))))))) +
                0.100000*np.tanh(np.minimum(((-2.0)), ((-2.0)))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + (np.minimum(((data["distmod"])), ((data["4__kurtosis_x"])))))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((((-3.0) / 2.0))), ((np.minimum(((data["4__kurtosis_x"])), ((data["3__skewness_x"]))))))) +
                0.100000*np.tanh(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"]))) * 2.0)) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((data["distmod"]) * 2.0)))))), ((data["3__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(((data["flux_ratio_sq_skew"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_dif2"])), ((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((data["flux_min"]) - (data["hostgal_photoz"]))))))))) - (data["detected_flux_err_max"]))) * 2.0)) +
                0.100000*np.tanh(((((data["distmod"]) + (data["distmod"]))) + (((data["distmod"]) + (data["flux_min"]))))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) * 2.0))), ((np.minimum(((((((np.minimum(((data["distmod"])), ((data["flux_by_flux_ratio_sq_skew"])))) * 2.0)) * 2.0))), ((data["distmod"]))))))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((data["distmod"])))) + (data["flux_min"]))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((np.minimum(((data["flux_by_flux_ratio_sq_skew"])), ((((((data["distmod"]) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_by_flux_ratio_sq_skew"]) + (np.minimum(((data["2__kurtosis_x"])), ((data["distmod"])))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["5__skewness_y"])), ((np.minimum(((data["flux_d0_pb3"])), ((data["detected_flux_min"])))))))), (((-1.0*((np.minimum(((data["flux_std"])), ((data["3__kurtosis_x"])))))))))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (((data["4__kurtosis_x"]) + (data["distmod"]))))) * 2.0)) + (((((data["detected_flux_max"]) / 2.0)) + (data["distmod"]))))) +
                0.100000*np.tanh(((np.minimum(((data["3__kurtosis_x"])), ((((data["distmod"]) + (((((data["3__skewness_x"]) + (data["2__kurtosis_x"]))) - (data["detected_flux_err_std"])))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["3__kurtosis_x"]) - (data["detected_flux_err_max"]))) - (0.367879))) - (data["flux_d0_pb3"]))) +
                0.100000*np.tanh(((np.minimum(((((data["flux_d1_pb2"]) + (data["4__kurtosis_x"])))), ((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])))))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) + ((((((((data["4__kurtosis_x"]) * 2.0)) + (data["distmod"]))/2.0)) + (((data["distmod"]) / 2.0)))))) + (data["2__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["3__kurtosis_x"])), ((data["flux_d0_pb3"]))))), ((np.minimum(((data["3__kurtosis_x"])), ((data["flux_median"]))))))) +
                0.100000*np.tanh(((((((data["detected_flux_min"]) - (data["detected_flux_err_median"]))) * 2.0)) + (((data["4__kurtosis_x"]) + (data["mjd_diff"]))))) +
                0.100000*np.tanh(np.minimum(((((((data["distmod"]) - (data["hostgal_photoz"]))) + (((data["distmod"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])))))), ((((data["flux_by_flux_ratio_sq_skew"]) + (data["distmod"])))))) +
                0.100000*np.tanh((((((((data["flux_min"]) + (((data["flux_min"]) * 2.0)))/2.0)) * 2.0)) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(((((((((data["distmod"]) * 2.0)) + (data["flux_d0_pb2"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb2"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - ((((((data["flux_d1_pb2"]) - (data["0__skewness_y"]))) < (data["flux_dif2"]))*1.)))) +
                0.100000*np.tanh((((((((((6.0)) / 2.0)) + (data["flux_min"]))) - (data["4__kurtosis_x"]))) - (data["hostgal_photoz"]))) +
                0.100000*np.tanh((((((-1.0*((data["2__fft_coefficient__coeff_1__attr__abs__x"])))) + (((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["flux_by_flux_ratio_sq_skew"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["distmod"])), ((np.minimum(((data["distmod"])), ((data["flux_dif2"]))))))) +
                0.100000*np.tanh(((data["4__kurtosis_x"]) + (data["flux_d0_pb1"]))) +
                0.100000*np.tanh(np.minimum(((((data["distmod"]) / 2.0))), ((np.where(data["distmod"]<0, (((data["distmod"]) + ((-1.0*((data["flux_dif2"])))))/2.0), data["1__kurtosis_x"] ))))) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (data["flux_d0_pb2"]))) +
                0.100000*np.tanh(np.where(((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) > -1, ((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (((data["flux_by_flux_ratio_sq_skew"]) - (data["hostgal_photoz"])))))), data["flux_median"] )) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (((data["flux_d0_pb2"]) + (((data["distmod"]) * 2.0)))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["flux_dif2"])), ((((data["distmod"]) + (data["distmod"]))))))), ((data["distmod"])))) +
                0.100000*np.tanh((((((data["4__skewness_x"]) + (data["flux_min"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_d0_pb2"]))))/2.0)) +
                0.100000*np.tanh((((data["detected_flux_min"]) + (((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_d1_pb1"]))))/2.0)) +
                0.100000*np.tanh(((((((((data["3__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["hostgal_photoz"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(((np.tanh((((((((data["distmod"]) * 2.0)) + (data["flux_d0_pb1"]))) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["5__kurtosis_x"])), ((np.minimum(((data["1__skewness_x"])), ((((data["5__kurtosis_x"]) * 2.0)))))))) +
                0.100000*np.tanh(((data["1__skewness_y"]) + (data["distmod"]))) +
                0.100000*np.tanh((((((data["5__kurtosis_x"]) > ((((data["hostgal_photoz"]) < (data["hostgal_photoz"]))*1.)))*1.)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.where(data["2__kurtosis_x"]>0, data["flux_d0_pb2"], data["2__fft_coefficient__coeff_0__attr__abs__x"] )) * (data["2__kurtosis_x"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((np.where(np.minimum(((data["flux_d0_pb2"])), ((data["detected_flux_min"]))) > -1, data["5__kurtosis_x"], data["flux_ratio_sq_skew"] ))), ((data["flux_d0_pb2"])))) +
                0.100000*np.tanh(((np.minimum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), ((data["2__fft_coefficient__coeff_0__attr__abs__x"])))) * (data["2__kurtosis_x"]))) +
                0.100000*np.tanh((((data["flux_d0_pb3"]) + (data["flux_d0_pb3"]))/2.0)) +
                0.100000*np.tanh(np.where(data["distmod"]>0, np.where(data["distmod"] > -1, 2.0, data["distmod"] ), -2.0 )) +
                0.100000*np.tanh(((((((-1.0*((data["4__skewness_x"])))) < (np.tanh((data["hostgal_photoz"]))))*1.)) + ((-1.0*((data["hostgal_photoz"])))))) +
                0.100000*np.tanh((((data["flux_d0_pb2"]) + ((((np.tanh((data["1__skewness_y"]))) + (data["1__skewness_y"]))/2.0)))/2.0)) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, np.where(2.0<0, data["flux_err_median"], data["detected_mean"] ), ((data["2__kurtosis_y"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"])) )) +
                0.100000*np.tanh((((data["flux_by_flux_ratio_sq_skew"]) + (np.where(data["distmod"]>0, data["4__kurtosis_x"], data["distmod"] )))/2.0)) +
                0.100000*np.tanh(((np.where(((data["flux_median"]) + (data["flux_median"]))>0, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["0__fft_coefficient__coeff_0__attr__abs__x"] )) * (data["3__kurtosis_x"]))) +
                0.100000*np.tanh(((data["detected_flux_w_mean"]) * (np.where(data["detected_flux_err_std"]>0, data["ddf"], data["flux_d0_pb4"] )))) +
                0.100000*np.tanh((((((data["distmod"]) + (data["detected_flux_min"]))) > (data["flux_d1_pb5"]))*1.)) +
                0.100000*np.tanh(np.where(data["distmod"]<0, np.where(data["flux_median"] > -1, data["flux_d0_pb1"], data["flux_w_mean"] ), ((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"])) )) +
                0.100000*np.tanh(np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_median"], ((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, np.maximum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))), data["detected_flux_w_mean"] )) + (data["detected_flux_by_flux_ratio_sq_sum"])) )) +
                0.100000*np.tanh(np.where(((data["detected_flux_err_median"]) * (data["3__fft_coefficient__coeff_0__attr__abs__x"]))<0, (((data["flux_d0_pb3"]) + (data["flux_diff"]))/2.0), (((data["flux_d0_pb3"]) < (data["1__skewness_y"]))*1.) )) +
                0.100000*np.tanh(((np.where(data["flux_median"]>0, data["flux_by_flux_ratio_sq_skew"], ((data["distmod"]) + (((data["distmod"]) + (data["detected_flux_w_mean"])))) )) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d0_pb2"]) + (((data["flux_d0_pb2"]) * 2.0)))) + (((data["1__skewness_y"]) + (data["detected_flux_err_min"]))))) +
                0.100000*np.tanh(np.minimum(((((data["5__kurtosis_x"]) / 2.0))), ((((np.tanh((data["flux_by_flux_ratio_sq_skew"]))) * (np.tanh((data["flux_median"])))))))) +
                0.100000*np.tanh(np.where(data["mjd_size"] > -1, (((data["flux_d0_pb5"]) < ((((((data["0__kurtosis_y"]) / 2.0)) > (data["flux_d0_pb5"]))*1.)))*1.), data["flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(((((((((((data["detected_flux_max"]) + (data["distmod"]))) + (data["distmod"]))) * 2.0)) + (data["distmod"]))) * 2.0)) +
                0.100000*np.tanh((((data["detected_flux_std"]) + (((data["2__skewness_y"]) + (np.minimum(((data["3__kurtosis_y"])), ((data["flux_d0_pb1"])))))))/2.0)) +
                0.100000*np.tanh((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) > (data["flux_w_mean"]))*1.)) +
                0.100000*np.tanh(((data["flux_dif2"]) * (((data["detected_flux_median"]) - (((data["detected_flux_median"]) * (((data["4__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))))))))) +
                0.100000*np.tanh(((((data["4__kurtosis_x"]) * (np.where(data["0__skewness_x"]<0, data["detected_flux_by_flux_ratio_sq_skew"], data["flux_d0_pb4"] )))) * (((data["flux_d1_pb1"]) * 2.0)))) +
                0.100000*np.tanh(np.where((((data["3__fft_coefficient__coeff_0__attr__abs__y"]) > (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.)<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * (data["3__kurtosis_x"])) )) +
                0.100000*np.tanh(((np.where((((data["flux_d0_pb3"]) > (data["flux_d0_pb5"]))*1.)>0, (((data["flux_d0_pb3"]) > (data["flux_dif3"]))*1.), ((data["flux_dif3"]) - (data["flux_d0_pb5"])) )) * 2.0)) +
                0.100000*np.tanh(np.maximum(((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["detected_flux_by_flux_ratio_sq_skew"])))), ((data["flux_d0_pb3"])))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, data["flux_median"], data["detected_flux_err_min"] )) +
                0.100000*np.tanh(((data["mjd_size"]) + (np.maximum(((data["detected_flux_err_mean"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"])))))) +
                0.100000*np.tanh(((((((((data["distmod"]) + (((((data["hostgal_photoz"]) + (data["detected_flux_max"]))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(np.maximum(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))), ((data["flux_err_skew"]))) > -1, data["5__kurtosis_x"], data["detected_flux_err_median"] )) +
                0.100000*np.tanh(((np.where(data["3__kurtosis_x"] > -1, data["3__kurtosis_x"], ((((data["detected_mjd_size"]) + (data["3__kurtosis_x"]))) * 2.0) )) * (((data["detected_mjd_size"]) * 2.0)))) +
                0.100000*np.tanh(((data["detected_mjd_size"]) * (((((((((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) * 2.0)) * 2.0)) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh(((np.tanh((np.where(data["flux_median"] > -1, data["flux_median"], data["flux_median"] )))) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, np.minimum(((data["3__kurtosis_x"])), ((data["flux_d0_pb0"]))), np.where(data["detected_flux_min"]>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_dif3"] ) )) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * (np.where(data["mwebv"]<0, data["3__kurtosis_x"], data["3__fft_coefficient__coeff_1__attr__abs__y"] )))) / 2.0)) +
                0.100000*np.tanh(np.where((((data["distmod"]) + (data["flux_err_skew"]))/2.0)>0, data["1__skewness_y"], data["hostgal_photoz_err"] )) +
                0.100000*np.tanh(((((((((((((data["detected_flux_max"]) + (data["hostgal_photoz"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]<0, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((((data["0__skewness_x"]) > (data["hostgal_photoz"]))*1.)) > (data["flux_d1_pb0"]))*1.)) +
                0.100000*np.tanh(((((((data["flux_d0_pb0"]) * 2.0)) * (((data["3__kurtosis_x"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((data["detected_flux_median"]) * (data["flux_skew"]))) +
                0.100000*np.tanh(np.where(data["flux_median"]>0, np.minimum(((data["5__kurtosis_x"])), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))), np.minimum(((data["flux_d0_pb2"])), ((np.minimum(((data["flux_d1_pb2"])), ((data["flux_d0_pb2"])))))) )) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"] > -1, ((data["flux_by_flux_ratio_sq_skew"]) * (data["detected_flux_ratio_sq_skew"])), np.where(data["flux_ratio_sq_sum"] > -1, ((data["flux_by_flux_ratio_sq_skew"]) * (data["flux_ratio_sq_skew"])), data["0__skewness_x"] ) )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"] > -1, ((data["detected_flux_err_mean"]) * (data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * (data["5__fft_coefficient__coeff_0__attr__abs__x"])) )) +
                0.100000*np.tanh(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["3__skewness_x"]))))) +
                0.100000*np.tanh((((data["flux_median"]) > (data["detected_flux_mean"]))*1.)) +
                0.100000*np.tanh(((((((data["flux_d0_pb2"]) + (((data["flux_median"]) + (data["flux_median"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_median"]))))) + (data["flux_median"]))) +
                0.100000*np.tanh(np.where(data["mjd_diff"] > -1, np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["4__skewness_x"], data["detected_flux_min"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_min"]<0, np.where(data["detected_flux_err_min"]>0, data["flux_max"], data["mjd_size"] ), data["mwebv"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, ((((data["distmod"]) * 2.0)) * 2.0), data["5__skewness_x"] )) +
                0.100000*np.tanh(((np.where(((data["3__skewness_x"]) * (data["detected_flux_by_flux_ratio_sq_skew"])) > -1, data["flux_err_skew"], data["detected_flux_by_flux_ratio_sq_skew"] )) + (((data["detected_flux_by_flux_ratio_sq_skew"]) * (data["3__kurtosis_x"]))))) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, np.maximum(((data["distmod"])), ((data["3__kurtosis_y"]))), np.tanh((data["flux_d0_pb2"])) )) +
                0.100000*np.tanh((((((((data["flux_d0_pb4"]) / 2.0)) + (data["2__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) * (data["flux_max"]))) +
                0.100000*np.tanh(np.where(np.where(data["flux_by_flux_ratio_sq_skew"]>0, data["flux_d1_pb1"], data["2__skewness_y"] )>0, data["flux_by_flux_ratio_sq_skew"], (((((data["flux_err_max"]) + (((data["2__kurtosis_y"]) / 2.0)))/2.0)) * 2.0) )) +
                0.100000*np.tanh(np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((((data["hostgal_photoz_err"]) * (data["flux_diff"])))), (((((data["flux_diff"]) + (data["flux_diff"]))/2.0))))))))))) +
                0.100000*np.tanh(np.where(data["flux_min"] > -1, np.where(data["flux_d0_pb0"]>0, data["flux_skew"], data["flux_d0_pb0"] ), data["detected_flux_by_flux_ratio_sq_skew"] )) +
                0.100000*np.tanh(np.where(data["flux_by_flux_ratio_sq_skew"] > -1, np.where((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) < (data["5__fft_coefficient__coeff_1__attr__abs__x"]))*1.)>0, data["mjd_diff"], data["mjd_size"] ), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((((np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]>0, data["flux_median"], data["detected_flux_err_std"] )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["flux_d0_pb2"]) + (((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))))) +
                0.100000*np.tanh((((np.where(data["detected_mjd_diff"] > -1, data["detected_flux_dif2"], data["detected_flux_dif2"] )) > (data["detected_mjd_diff"]))*1.)) +
                0.100000*np.tanh(np.where(((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["5__fft_coefficient__coeff_1__attr__abs__y"])) > -1, np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], -2.0 ), -2.0 )) +
                0.100000*np.tanh(np.where(((data["flux_diff"]) + (((data["distmod"]) + (data["distmod"]))))<0, ((data["distmod"]) + (data["hostgal_photoz_err"])), 2.718282 )) +
                0.100000*np.tanh(((data["4__skewness_x"]) * (((np.where(data["3__kurtosis_x"]>0, data["3__kurtosis_x"], ((data["3__kurtosis_x"]) * (data["flux_ratio_sq_sum"])) )) * (data["flux_ratio_sq_sum"]))))) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["flux_err_mean"]>0, data["detected_flux_min"], data["flux_ratio_sq_sum"] ), data["2__kurtosis_y"] )) +
                0.100000*np.tanh((((((((data["flux_d0_pb4"]) < (np.where(data["flux_d0_pb2"] > -1, data["flux_d0_pb2"], data["flux_d0_pb4"] )))*1.)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"]<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], (-1.0*((np.where(data["hostgal_photoz_err"]<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )))) )) +
                0.100000*np.tanh(((data["4__skewness_x"]) + (np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], data["distmod"] )))) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["1__kurtosis_y"], data["flux_err_mean"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, data["distmod"], np.where(data["5__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])), data["1__fft_coefficient__coeff_1__attr__abs__x"] ) )))

    def GP_class_92(self,data):
        return (-1.730312 +
                0.100000*np.tanh(((data["flux_err_min"]) + (((((data["flux_err_min"]) + (-3.0))) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) +
                0.100000*np.tanh(((((data["detected_mean"]) + (((data["detected_mean"]) + (-2.0))))) + (-2.0))) +
                0.100000*np.tanh(((((np.minimum(((((((data["flux_err_min"]) + (data["detected_mean"]))) * 2.0))), ((data["detected_mean"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((((((np.minimum(((np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["flux_err_min"]))))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(np.where(((data["4__kurtosis_x"]) * 2.0) > -1, (-1.0*((data["4__kurtosis_x"]))), ((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)) * 2.0) )) +
                0.100000*np.tanh((((-1.0*((data["5__kurtosis_x"])))) * 2.0)) +
                0.100000*np.tanh(np.minimum(((data["detected_mean"])), ((((np.minimum(((np.minimum(((((data["flux_err_min"]) * 2.0))), ((data["detected_mean"]))))), ((data["flux_err_min"])))) * 2.0))))) +
                0.100000*np.tanh((((((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((data["1__skewness_y"])))) + (-1.0))/2.0)) + (data["flux_err_min"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (-2.0))) + (((data["0__fft_coefficient__coeff_1__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))))) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
                0.100000*np.tanh(((np.minimum(((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) * 2.0))), ((((data["flux_err_min"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((data["5__kurtosis_x"]) + (((((((-1.0) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["4__kurtosis_y"]))))) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.where(data["3__kurtosis_x"] > -1, -2.0, np.where(data["3__kurtosis_x"] > -1, data["detected_flux_diff"], data["detected_flux_std"] ) ), data["0__fft_coefficient__coeff_1__attr__abs__y"] )) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__kurtosis_y"]))) * 2.0)) - (np.tanh((data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
                0.100000*np.tanh((-1.0*((np.where(np.minimum(((data["3__skewness_x"])), ((data["3__kurtosis_x"]))) > -1, np.where(data["3__kurtosis_x"] > -1, 3.141593, data["3__kurtosis_x"] ), data["3__kurtosis_x"] ))))) +
                0.100000*np.tanh(np.where(-3.0<0, np.where(data["4__kurtosis_x"] > -1, ((-3.0) - (-2.0)), data["detected_mean"] ), data["flux_max"] )) +
                0.100000*np.tanh(((((np.minimum(((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_diff"])))), ((-3.0)))) + (np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["detected_flux_max"])))))) - (data["5__kurtosis_x"]))) +
                0.100000*np.tanh((-1.0*((np.where(np.where(data["flux_err_max"] > -1, data["4__kurtosis_x"], 2.718282 ) > -1, 2.0, data["5__kurtosis_x"] ))))) +
                0.100000*np.tanh(((np.minimum(((data["detected_flux_max"])), ((np.minimum(((data["detected_flux_err_min"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"]))))))) * 2.0)) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["3__kurtosis_x"]))))), ((((((data["flux_err_min"]) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(((((((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))), ((data["detected_flux_err_min"])))) * 2.0)) * 2.0)) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["detected_flux_max"])), ((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))))))), ((data["flux_err_min"])))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"]<0, np.where(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)>0, data["flux_err_min"], data["1__fft_coefficient__coeff_1__attr__abs__x"] ), ((data["flux_err_min"]) * 2.0) )) * 2.0)) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, -3.0, np.where(data["3__kurtosis_y"] > -1, -3.0, ((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (-1.0)) ) )) +
                0.100000*np.tanh(np.where(data["3__kurtosis_x"] > -1, np.minimum(((data["5__kurtosis_x"])), ((-3.0))), np.where(2.0 > -1, data["3__fft_coefficient__coeff_1__attr__abs__x"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ) )) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, np.where(data["4__kurtosis_x"] > -1, -2.0, data["0__fft_coefficient__coeff_1__attr__abs__x"] ), data["detected_flux_std"] )) +
                0.100000*np.tanh(np.where(data["3__skewness_x"]<0, data["flux_diff"], np.minimum(((-2.0)), ((np.minimum(((np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, data["flux_diff"], data["3__skewness_x"] ))), ((data["distmod"])))))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__y"])), ((((data["flux_max"]) * 2.0)))))))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, np.where(data["hostgal_photoz"] > -1, -3.0, data["0__fft_coefficient__coeff_0__attr__abs__x"] ), data["0__fft_coefficient__coeff_0__attr__abs__y"] )) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, -3.0, (-1.0*((data["5__kurtosis_y"]))) )) +
                0.100000*np.tanh(np.minimum(((np.minimum(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (((data["flux_d1_pb2"]) * 2.0))))), ((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((data["2__fft_coefficient__coeff_1__attr__abs__y"])))))))), ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0))))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((data["4__kurtosis_x"])), ((data["detected_flux_min"])))))) - (data["1__kurtosis_y"]))) +
                0.100000*np.tanh(np.minimum(((((data["hostgal_photoz_err"]) + (np.minimum((((((data["flux_err_min"]) + (data["flux_err_min"]))/2.0))), ((np.minimum(((data["detected_flux_err_skew"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"])))))))))), ((data["detected_flux_max"])))) +
                0.100000*np.tanh(np.where(data["1__skewness_y"]>0, data["1__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["5__kurtosis_x"] > -1, -3.0, data["flux_err_min"] ) )) +
                0.100000*np.tanh(np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (np.maximum(((((data["detected_mean"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((1.0))))))), ((data["flux_err_min"])))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((np.minimum(((data["detected_mean"])), ((data["detected_flux_max"]))))), ((data["flux_err_skew"])))))))), ((data["detected_flux_err_min"])))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["4__kurtosis_y"]))) + (((data["0__fft_coefficient__coeff_0__attr__abs__x"]) - (data["flux_dif3"]))))) +
                0.100000*np.tanh(np.minimum(((data["detected_flux_max"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh(np.where(data["4__skewness_x"]<0, np.where(data["detected_flux_max"]<0, data["detected_flux_err_min"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ), -2.0 )) +
                0.100000*np.tanh(((((((-2.0) + (data["detected_flux_err_min"]))) + ((((data["2__skewness_x"]) + (data["1__skewness_y"]))/2.0)))) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((np.minimum(((data["flux_err_min"])), ((((data["detected_flux_max"]) + (data["2__skewness_y"]))))))))) +
                0.100000*np.tanh(np.minimum(((data["flux_err_min"])), ((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["4__kurtosis_x"])))))) +
                0.100000*np.tanh((((((-1.0*((((np.where(data["4__kurtosis_x"] > -1, data["5__kurtosis_x"], 1.0 )) * (data["1__fft_coefficient__coeff_0__attr__abs__x"])))))) - (data["5__kurtosis_y"]))) - (data["flux_min"]))) +
                0.100000*np.tanh(((data["flux_dif3"]) + ((-1.0*((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0)) + ((((data["flux_err_median"]) > (data["2__fft_coefficient__coeff_1__attr__abs__y"]))*1.))))))))) +
                0.100000*np.tanh(((data["flux_err_min"]) + (data["detected_flux_err_min"]))) +
                0.100000*np.tanh(((((data["detected_mean"]) - (np.where(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0) > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )))) * 2.0)) +
                0.100000*np.tanh(np.where(data["distmod"] > -1, -2.0, data["1__kurtosis_y"] )) +
                0.100000*np.tanh(np.where((-1.0*((((data["detected_flux_dif2"]) / 2.0)))) > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["flux_max"]<0, (-1.0*(((-1.0*((data["2__fft_coefficient__coeff_0__attr__abs__x"])))))), np.where(data["4__skewness_x"]<0, (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__skewness_y"]))/2.0), -3.0 ) )) +
                0.100000*np.tanh((((((((np.minimum(((data["1__skewness_y"])), ((data["0__fft_coefficient__coeff_1__attr__abs__x"])))) + (data["detected_flux_max"]))/2.0)) - (data["2__kurtosis_y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(np.minimum(((((data["flux_max"]) * (data["1__skewness_y"])))), ((data["detected_flux_std"])))>0, data["0__fft_coefficient__coeff_1__attr__abs__x"], np.minimum(((data["detected_flux_max"])), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) )) +
                0.100000*np.tanh(((data["flux_err_skew"]) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["flux_err_skew"]>0, data["flux_err_skew"], -3.0 )) +
                0.100000*np.tanh(((((data["detected_flux_by_flux_ratio_sq_skew"]) * 2.0)) - (data["3__kurtosis_x"]))) +
                0.100000*np.tanh((-1.0*((np.where(((data["detected_flux_min"]) - (data["5__kurtosis_x"]))>0, np.where(data["detected_flux_min"]<0, data["detected_flux_skew"], data["detected_flux_min"] ), data["5__kurtosis_x"] ))))) +
                0.100000*np.tanh(((data["ddf"]) - (data["4__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, -2.0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["detected_flux_ratio_sq_sum"] ) )) +
                0.100000*np.tanh(((np.where(data["0__skewness_x"]>0, data["1__fft_coefficient__coeff_0__attr__abs__x"], ((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["4__fft_coefficient__coeff_0__attr__abs__y"])) )) - (data["4__fft_coefficient__coeff_0__attr__abs__y"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["flux_ratio_sq_sum"] > -1, data["detected_flux_err_skew"], data["2__skewness_x"] ))), ((data["detected_flux_err_min"])))) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) - (data["3__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((np.where(data["5__kurtosis_x"] > -1, np.minimum(((3.0)), ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["mwebv"]))))), ((data["detected_mjd_size"]) / 2.0) ))))) +
                0.100000*np.tanh(np.minimum(((data["flux_dif3"])), ((np.minimum(((data["flux_dif3"])), ((data["detected_flux_err_std"]))))))) +
                0.100000*np.tanh(((((((data["1__skewness_y"]) + (np.where(data["1__skewness_y"] > -1, data["flux_err_std"], data["1__fft_coefficient__coeff_0__attr__abs__x"] )))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["1__skewness_y"]))) +
                0.100000*np.tanh((((((data["flux_d0_pb5"]) < (np.minimum(((data["flux_max"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))))*1.)) / 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["flux_err_median"])), ((((((((data["1__kurtosis_y"]) < (data["2__fft_coefficient__coeff_1__attr__abs__x"]))*1.)) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))/2.0))))) - (data["4__skewness_y"]))) +
                0.100000*np.tanh(((((data["2__skewness_y"]) * (((data["1__fft_coefficient__coeff_1__attr__abs__y"]) * (data["2__skewness_y"]))))) - ((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((((data["flux_d1_pb3"]) / 2.0)) * 2.0)))/2.0)))) +
                0.100000*np.tanh(np.where(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where((9.86017036437988281)>0, data["flux_d0_pb3"], data["flux_dif3"] ), (-1.0*((data["4__skewness_y"]))) ) > -1, data["detected_flux_err_skew"], data["2__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.minimum(((((-1.0) / 2.0))), ((np.tanh((data["detected_mean"])))))) +
                0.100000*np.tanh((((((np.tanh((data["1__skewness_y"]))) - (data["flux_d1_pb3"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) +
                0.100000*np.tanh(((np.minimum(((np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["2__skewness_y"])))) * 2.0)) +
                0.100000*np.tanh(np.where(data["flux_d1_pb5"] > -1, np.where(data["detected_flux_min"] > -1, data["detected_flux_min"], data["0__skewness_x"] ), data["detected_flux_err_skew"] )) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], ((data["flux_dif3"]) + (data["mwebv"])) )) +
                0.100000*np.tanh(((((np.where(data["4__skewness_x"]<0, data["1__skewness_y"], (((-1.0*((((data["2__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0))))) / 2.0) )) * 2.0)) / 2.0)) +
                0.100000*np.tanh(((data["detected_mean"]) - (np.where(data["flux_dif3"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__x"], (((data["3__kurtosis_x"]) < (data["2__kurtosis_y"]))*1.) )))) +
                0.100000*np.tanh(((np.where(data["flux_d0_pb4"]>0, np.minimum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_dif3"]))), data["flux_d0_pb4"] )) - (data["detected_flux_mean"]))) +
                0.100000*np.tanh(np.where(np.maximum((((((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["detected_flux_by_flux_ratio_sq_sum"])))), ((data["2__fft_coefficient__coeff_1__attr__abs__y"]))) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.where(data["flux_d1_pb1"]>0, ((data["detected_flux_diff"]) + (np.where(data["flux_dif3"]>0, np.maximum(((data["1__fft_coefficient__coeff_0__attr__abs__y"])), ((data["detected_flux_err_skew"]))), -3.0 ))), data["flux_d1_pb1"] )) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, (((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["5__skewness_x"]))/2.0), data["mwebv"] )) +
                0.100000*np.tanh(((np.minimum(((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (data["1__kurtosis_x"])))), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)))) +
                0.100000*np.tanh(np.where((-1.0*((np.tanh((data["detected_flux_std"])))))<0, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(np.where((((np.where(data["flux_ratio_sq_sum"] > -1, data["3__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_by_flux_ratio_sq_skew"] )) + (data["detected_flux_by_flux_ratio_sq_skew"]))/2.0)<0, data["flux_d1_pb4"], data["mwebv"] )) +
                0.100000*np.tanh(((((data["detected_flux_err_mean"]) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__skewness_x"]))) +
                0.100000*np.tanh(np.minimum(((np.where(data["1__skewness_y"]<0, np.where(data["detected_mean"]>0, data["1__skewness_y"], data["1__skewness_y"] ), data["1__fft_coefficient__coeff_1__attr__abs__y"] ))), ((data["flux_d1_pb1"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_max"]>0, ((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__skewness_y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])), data["2__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((data["detected_flux_err_skew"]) - (np.where(((data["flux_diff"]) / 2.0) > -1, ((data["flux_diff"]) - (data["flux_d1_pb4"])), ((data["flux_diff"]) / 2.0) )))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_0__attr__abs__y"]) + (np.tanh((data["2__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, data["flux_by_flux_ratio_sq_skew"], np.where(data["5__kurtosis_x"] > -1, -1.0, data["detected_flux_diff"] ) )) +
                0.100000*np.tanh(((data["detected_flux_ratio_sq_skew"]) * (2.0))) +
                0.100000*np.tanh(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (np.maximum(((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d0_pb3"])))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) +
                0.100000*np.tanh(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, np.minimum(((data["flux_err_skew"])), ((((data["detected_flux_err_max"]) - (data["2__kurtosis_y"]))))), data["flux_ratio_sq_sum"] )) +
                0.100000*np.tanh(np.minimum(((data["0__kurtosis_y"])), ((data["1__fft_coefficient__coeff_1__attr__abs__y"])))) +
                0.100000*np.tanh((-1.0*((np.where(data["flux_ratio_sq_skew"]>0, np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__y"])), (-1.0*((data["0__fft_coefficient__coeff_1__attr__abs__x"]))) ))))) +
                0.100000*np.tanh(np.where((((data["flux_err_skew"]) > (((data["detected_flux_err_skew"]) / 2.0)))*1.)<0, data["1__fft_coefficient__coeff_0__attr__abs__x"], np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((data["detected_flux_err_skew"]))) )) +
                0.100000*np.tanh(np.minimum(((data["flux_err_max"])), ((((data["flux_err_skew"]) * (((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (-3.0)))))))) +
                0.100000*np.tanh((((((2.718282) / 2.0)) < (np.where(data["mwebv"]<0, data["flux_ratio_sq_sum"], (-1.0*(((-1.0*((data["mjd_size"])))))) )))*1.)) +
                0.100000*np.tanh((((data["flux_err_skew"]) + (data["detected_mean"]))/2.0)) +
                0.100000*np.tanh(np.where(data["0__fft_coefficient__coeff_1__attr__abs__y"] > -1, np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), (((((np.where(data["flux_max"]<0, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["flux_dif3"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)))), -3.0 )) +
                0.100000*np.tanh(((np.minimum(((data["detected_mean"])), (((((((data["detected_flux_err_min"]) + (data["mwebv"]))/2.0)) / 2.0))))) / 2.0)) +
                0.100000*np.tanh(np.where(data["5__kurtosis_x"] > -1, (-1.0*(((((data["5__skewness_x"]) < (data["5__kurtosis_x"]))*1.)))), (((data["flux_d1_pb5"]) < (data["5__fft_coefficient__coeff_1__attr__abs__y"]))*1.) )) +
                0.100000*np.tanh((-1.0*((np.tanh(((-1.0*(((((((-1.0*((((np.tanh((data["1__skewness_y"]))) / 2.0))))) * (data["3__skewness_x"]))) * 2.0)))))))))) +
                0.100000*np.tanh(np.minimum(((data["1__skewness_x"])), ((data["0__kurtosis_y"])))) +
                0.100000*np.tanh(((((((data["flux_std"]) > (np.minimum(((data["5__skewness_y"])), ((data["flux_diff"])))))*1.)) < (np.minimum(((data["flux_err_min"])), ((data["1__skewness_x"])))))*1.)) +
                0.100000*np.tanh(((data["3__fft_coefficient__coeff_1__attr__abs__y"]) - ((-1.0*((data["4__kurtosis_y"])))))) +
                0.100000*np.tanh(((((((-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__y"])))) + (data["4__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) - (((data["1__kurtosis_x"]) * 2.0)))) +
                0.100000*np.tanh(np.where((2.0)>0, (((((data["flux_d1_pb3"]) < (data["3__kurtosis_y"]))*1.)) / 2.0), (((((data["flux_err_std"]) / 2.0)) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) )) +
                0.100000*np.tanh(((((np.tanh((np.tanh((((data["detected_flux_mean"]) + (data["detected_flux_by_flux_ratio_sq_skew"]))))))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh((((np.minimum(((0.0)), ((data["flux_err_mean"])))) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) +
                0.100000*np.tanh(((np.where(data["detected_flux_err_skew"]<0, data["1__skewness_y"], data["flux_d1_pb0"] )) - (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.tanh((((data["1__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_median"]))))) +
                0.100000*np.tanh((-1.0*((np.where(data["flux_max"] > -1, ((((data["mjd_diff"]) - (np.tanh((data["flux_max"]))))) / 2.0), data["flux_dif3"] ))))) +
                0.100000*np.tanh(np.minimum((((((data["0__skewness_y"]) < (((np.tanh((data["flux_dif3"]))) * 2.0)))*1.))), ((np.minimum(((-1.0)), ((data["1__skewness_x"]))))))) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]>0, 2.0, data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]<0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0), np.minimum(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))), ((data["1__fft_coefficient__coeff_0__attr__abs__x"]))) )) +
                0.100000*np.tanh(np.where(data["detected_flux_min"]<0, np.where(data["4__kurtosis_x"]<0, data["detected_flux_by_flux_ratio_sq_skew"], ((-2.0) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])) ), ((-2.0) - (data["detected_flux_min"])) )) +
                0.100000*np.tanh(((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__x"] )) / 2.0)) +
                0.100000*np.tanh(np.minimum((((-1.0*((data["3__fft_coefficient__coeff_1__attr__abs__y"]))))), ((data["flux_d0_pb4"])))) +
                0.100000*np.tanh(np.where(data["detected_flux_by_flux_ratio_sq_skew"] > -1, data["flux_max"], ((data["flux_max"]) - (((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["1__skewness_y"]))) / 2.0))) )) +
                0.100000*np.tanh(np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), (((((3.0)) * 2.0))))) +
                0.100000*np.tanh(((((3.141593) - (((((data["flux_ratio_sq_sum"]) * 2.0)) - (data["mwebv"]))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_err_skew"]))) +
                0.100000*np.tanh(np.where((((data["1__skewness_y"]) < (data["flux_err_skew"]))*1.) > -1, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["hostgal_photoz"] )))

    def GP_class_95(self,data):
        return (-1.890339 +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) * 2.0)) + (np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) * 2.0)) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (data["hostgal_photoz"]))) + (data["distmod"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((((((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((data["distmod"]) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)) + (data["flux_w_mean"]))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["distmod"]) * 2.0)))) * 2.0)) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((((((data["distmod"]) + (((data["distmod"]) * 2.0)))) + (data["distmod"]))) + (((data["distmod"]) + (data["detected_flux_w_mean"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (((data["distmod"]) + (((data["hostgal_photoz"]) + (data["distmod"]))))))))) +
                0.100000*np.tanh(((((data["flux_d1_pb4"]) + (((data["detected_flux_min"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["distmod"]) + (((((np.minimum(((data["3__skewness_y"])), ((((data["flux_d1_pb0"]) * 2.0))))) / 2.0)) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) + (((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) * 2.0)))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (np.minimum(((((data["flux_d1_pb5"]) + (data["hostgal_photoz_err"])))), ((data["distmod"])))))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["flux_d0_pb5"]) + (((data["hostgal_photoz"]) + (data["flux_d0_pb5"]))))))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (data["distmod"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (((data["flux_median"]) + (data["hostgal_photoz"]))))) * 2.0)) +
                0.100000*np.tanh((((((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))))/2.0)) + (data["flux_d0_pb3"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_d0_pb5"])), ((np.minimum(((data["distmod"])), ((((data["flux_max"]) + (data["5__skewness_x"]))))))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["distmod"]) + ((((data["0__skewness_x"]) + (((data["distmod"]) * 2.0)))/2.0)))))) +
                0.100000*np.tanh(((((np.minimum(((((((data["hostgal_photoz"]) + (data["flux_d0_pb1"]))) * 2.0))), ((data["hostgal_photoz"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (((((np.minimum(((data["hostgal_photoz"])), ((data["5__skewness_x"])))) * 2.0)) + (data["flux_d0_pb5"]))))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((((data["flux_median"]) + (np.minimum(((np.minimum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))))), ((((((data["hostgal_photoz"]) * 2.0)) + (data["detected_flux_min"])))))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) + (((((data["distmod"]) * 2.0)) * 2.0))))), ((((data["flux_d0_pb5"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((((data["flux_d0_pb4"]) - (data["hostgal_photoz_err"]))) * 2.0)) + (((data["flux_d0_pb4"]) - (data["flux_d0_pb5"]))))) * 2.0)) +
                0.100000*np.tanh(((data["flux_d1_pb4"]) + (((data["detected_flux_min"]) + (data["4__skewness_x"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) * 2.0)) + (data["detected_flux_std"]))) +
                0.100000*np.tanh(((((data["distmod"]) - (((data["hostgal_photoz_err"]) * 2.0)))) + (data["0__skewness_x"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb5"])), ((data["hostgal_photoz"])))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((((data["hostgal_photoz"]) * 2.0)))))), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))))), ((((data["hostgal_photoz"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((np.minimum(((data["1__skewness_x"])), ((((((data["flux_d1_pb5"]) * 2.0)) + (((data["1__skewness_x"]) * 2.0)))))))))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__skewness_x"]) + (data["detected_flux_min"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((data["distmod"])), ((((data["0__skewness_x"]) + (data["0__skewness_x"])))))) + (((data["0__skewness_x"]) + (((data["0__skewness_x"]) * 2.0)))))) +
                0.100000*np.tanh(((data["distmod"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["detected_flux_median"]) + (data["hostgal_photoz"]))) + (data["flux_std"]))) + (((np.minimum(((data["5__skewness_x"])), ((data["hostgal_photoz"])))) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((np.minimum(((data["4__skewness_x"])), ((data["hostgal_photoz"])))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((((data["distmod"]) - (data["hostgal_photoz_err"])))), ((((data["hostgal_photoz"]) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["hostgal_photoz_err"]))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["detected_flux_min"]) + (data["hostgal_photoz"]))) + (data["flux_std"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((((data["flux_d0_pb5"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"]))) * 2.0)) +
                0.100000*np.tanh(((((((((data["hostgal_photoz"]) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) + (data["hostgal_photoz"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["5__skewness_x"])), ((((((data["0__skewness_x"]) * 2.0)) * 2.0))))) +
                0.100000*np.tanh(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["distmod"]) + (data["detected_flux_min"]))))) * 2.0)) +
                0.100000*np.tanh(((((np.minimum(((((((data["detected_flux_err_mean"]) + (data["hostgal_photoz"]))) + (data["hostgal_photoz"])))), ((data["flux_d0_pb5"])))) + (((data["hostgal_photoz"]) * 2.0)))) * 2.0)) +
                0.100000*np.tanh(((np.minimum(((np.minimum(((data["hostgal_photoz"])), ((((data["flux_d0_pb4"]) - (data["hostgal_photoz_err"]))))))), ((data["1__skewness_x"])))) * 2.0)) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz_err"]<0, data["distmod"], ((((data["hostgal_photoz"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"])) )) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["flux_max"]))) + (((((data["hostgal_photoz"]) + (data["flux_std"]))) + (((data["hostgal_photoz"]) + (-1.0))))))) +
                0.100000*np.tanh(((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (np.where(((data["2__kurtosis_x"]) / 2.0) > -1, data["hostgal_photoz_err"], (((((data["hostgal_photoz_err"]) * 2.0)) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))/2.0) )))) +
                0.100000*np.tanh(((((data["flux_d1_pb5"]) + (np.minimum(((data["1__skewness_x"])), ((np.minimum(((data["flux_d0_pb4"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["flux_skew"])))))))))))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (((data["hostgal_photoz"]) + (((np.minimum(((data["flux_diff"])), ((data["5__fft_coefficient__coeff_0__attr__abs__x"])))) * 2.0)))))) + (((data["hostgal_photoz"]) * 2.0)))) +
                0.100000*np.tanh(((np.minimum(((((np.minimum(((data["detected_flux_min"])), ((((np.tanh((data["hostgal_photoz"]))) - (data["hostgal_photoz_err"])))))) * 2.0))), ((data["hostgal_photoz"])))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__kurtosis_x"]))) - (data["2__kurtosis_x"]))) - (data["2__kurtosis_x"]))) +
                0.100000*np.tanh(((((np.minimum(((data["flux_skew"])), ((((((data["hostgal_photoz"]) * 2.0)) * 2.0))))) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["hostgal_photoz_err"]) + (data["hostgal_photoz"]))) - (((data["hostgal_photoz_err"]) * (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(((np.where(data["3__skewness_x"]<0, data["flux_dif3"], ((data["flux_d0_pb0"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"])) )) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((data["hostgal_photoz_err"]) + (((data["flux_d1_pb0"]) + (np.minimum(((data["hostgal_photoz"])), ((data["detected_mjd_diff"])))))))))) +
                0.100000*np.tanh(np.where(data["flux_dif2"]<0, data["flux_d1_pb4"], -3.0 )) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (((((((data["hostgal_photoz"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) + (data["hostgal_photoz"]))))) * 2.0)) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["hostgal_photoz_err"]))) - (data["1__skewness_y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((((np.minimum(((data["flux_median"])), ((((data["flux_d0_pb5"]) - (data["flux_skew"])))))) * 2.0))))) * 2.0)) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))) * 2.0)) + (0.0))) * 2.0)) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) +
                0.100000*np.tanh(((((data["2__kurtosis_y"]) - (data["hostgal_photoz_err"]))) * 2.0)) +
                0.100000*np.tanh(((((np.where(data["flux_skew"] > -1, data["flux_skew"], data["hostgal_photoz"] )) - (((data["hostgal_photoz_err"]) * (data["flux_skew"]))))) * (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["1__skewness_x"]) * 2.0))))) +
                0.100000*np.tanh(((((data["flux_diff"]) + (data["detected_flux_err_mean"]))) + (data["3__skewness_x"]))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (((((-1.0) + (((data["hostgal_photoz"]) * 2.0)))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0))) )) * 2.0)) +
                0.100000*np.tanh(((((((((data["detected_flux_median"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) / 2.0)) + (data["0__skewness_x"]))) +
                0.100000*np.tanh((((((data["hostgal_photoz_err"]) < (data["4__fft_coefficient__coeff_1__attr__abs__y"]))*1.)) - ((((data["hostgal_photoz_err"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["hostgal_photoz_err"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))/2.0)))) +
                0.100000*np.tanh(((data["hostgal_photoz"]) + (np.minimum(((((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))), ((data["flux_max"])))))) +
                0.100000*np.tanh(np.minimum(((np.tanh((np.minimum(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["hostgal_photoz_err"])))), ((data["flux_median"]))))))), ((data["flux_median"])))) +
                0.100000*np.tanh((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(data["5__fft_coefficient__coeff_0__attr__abs__y"]<0, data["1__kurtosis_x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )))/2.0)) - (data["hostgal_photoz_err"]))) + (data["detected_mjd_diff"]))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"]))))), ((np.where(data["flux_median"]<0, data["detected_mjd_diff"], np.minimum(((data["detected_mjd_diff"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"]))) ))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["detected_flux_mean"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(((np.minimum(((data["flux_d0_pb0"])), ((np.minimum(((((np.minimum(((data["detected_mjd_size"])), ((data["3__fft_coefficient__coeff_0__attr__abs__y"])))) / 2.0))), ((data["1__skewness_x"]))))))) * 2.0)) +
                0.100000*np.tanh(((((data["2__kurtosis_y"]) + ((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["hostgal_photoz"]))/2.0)))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((data["2__kurtosis_y"])), ((np.minimum(((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["flux_d0_pb3"]))))), ((((((data["flux_median"]) * 2.0)) * 2.0))))))))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_0__attr__abs__x"]) + (data["hostgal_photoz"]))) + (data["5__fft_coefficient__coeff_0__attr__abs__x"]))) + (((((((data["hostgal_photoz"]) * 2.0)) * 2.0)) + (data["hostgal_photoz_err"]))))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + ((((data["0__skewness_y"]) + (((data["detected_flux_err_mean"]) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["hostgal_photoz"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))))))))/2.0)))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"]>0, ((np.where(data["hostgal_photoz"] > -1, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_err_median"] )) - (data["hostgal_photoz_err"])), data["0__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["4__skewness_x"]))) +
                0.100000*np.tanh(((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))) +
                0.100000*np.tanh(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((((np.where(data["flux_skew"]<0, data["detected_flux_err_std"], data["hostgal_photoz"] )) + (data["4__skewness_y"]))) + (data["flux_d0_pb0"]))))) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (data["hostgal_photoz"]))) + (((((data["1__skewness_y"]) + (data["hostgal_photoz"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]>0, data["mjd_diff"], np.where(data["2__kurtosis_x"]>0, ((data["3__kurtosis_y"]) + (data["2__kurtosis_x"])), data["3__kurtosis_y"] ) )) +
                0.100000*np.tanh(np.where(data["1__skewness_x"]<0, data["1__skewness_x"], data["detected_flux_mean"] )) +
                0.100000*np.tanh(((data["flux_d1_pb3"]) + (((np.minimum(((data["2__kurtosis_y"])), ((data["detected_flux_max"])))) + (data["2__kurtosis_y"]))))) +
                0.100000*np.tanh(((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)) +
                0.100000*np.tanh(np.where(data["detected_flux_min"] > -1, ((((data["flux_min"]) / 2.0)) - (data["1__skewness_y"])), data["1__skewness_x"] )) +
                0.100000*np.tanh(np.minimum(((data["flux_median"])), ((((data["flux_skew"]) + (np.minimum(((np.minimum(((data["flux_median"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"])))))))), ((data["flux_median"]))))))))) +
                0.100000*np.tanh((((data["flux_d0_pb5"]) + (data["flux_skew"]))/2.0)) +
                0.100000*np.tanh(((((data["detected_mjd_diff"]) + (data["detected_flux_err_median"]))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]<0, data["flux_d1_pb1"], ((np.minimum(((data["detected_mjd_diff"])), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) - (data["detected_mean"])) )) +
                0.100000*np.tanh(np.where(data["detected_flux_ratio_sq_skew"]<0, (((((data["flux_d0_pb5"]) * 2.0)) + (data["1__skewness_x"]))/2.0), data["detected_mjd_diff"] )) +
                0.100000*np.tanh(np.where(data["hostgal_photoz_err"]<0, data["3__skewness_x"], np.where(data["hostgal_photoz"]<0, np.maximum(((data["hostgal_photoz"])), ((data["hostgal_photoz_err"]))), ((data["3__skewness_x"]) - (data["hostgal_photoz_err"])) ) )) +
                0.100000*np.tanh((((data["flux_median"]) + (((data["detected_flux_err_median"]) + (data["4__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) +
                0.100000*np.tanh(np.where(data["flux_dif3"]>0, np.where(np.where(data["flux_dif3"]>0, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_dif3"] )>0, data["flux_dif3"], data["flux_dif3"] ), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
                0.100000*np.tanh(((((((((((data["distmod"]) - (data["hostgal_photoz_err"]))) - (data["1__skewness_y"]))) - (data["1__skewness_y"]))) - (data["1__skewness_y"]))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((np.where(((data["3__kurtosis_y"]) * 2.0)<0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.tanh((((data["1__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0))) )) * 2.0)) * 2.0)) +
                0.100000*np.tanh(((((data["flux_d1_pb0"]) + (data["hostgal_photoz"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.minimum(((np.minimum(((data["detected_mjd_diff"])), ((data["2__kurtosis_y"]))))), ((data["2__kurtosis_y"])))) +
                0.100000*np.tanh(((data["detected_flux_err_std"]) + (data["flux_d1_pb5"]))) +
                0.100000*np.tanh(np.where((-1.0*((data["hostgal_photoz_err"]))) > -1, data["hostgal_photoz"], (-1.0*((((data["detected_flux_err_min"]) + (data["hostgal_photoz"]))))) )) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((((data["hostgal_photoz_err"]) + (((data["hostgal_photoz"]) + (((data["hostgal_photoz_err"]) + (data["flux_max"]))))))) + (data["hostgal_photoz"]))))) +
                0.100000*np.tanh(np.where(data["4__kurtosis_y"]>0, np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["flux_err_std"]>0, data["flux_err_min"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ), data["3__kurtosis_y"] ), data["flux_median"] )) +
                0.100000*np.tanh(((data["hostgal_photoz_err"]) + (((((data["flux_d1_pb0"]) - (data["hostgal_photoz_err"]))) - (data["hostgal_photoz_err"]))))) +
                0.100000*np.tanh(((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))) + (((data["detected_mean"]) + (((data["hostgal_photoz_err"]) + (data["detected_flux_skew"]))))))) +
                0.100000*np.tanh(np.where(data["2__kurtosis_x"]>0, np.tanh((np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - ((-1.0*((data["4__fft_coefficient__coeff_1__attr__abs__y"])))))>0, data["mjd_diff"], data["1__kurtosis_y"] ))), data["4__fft_coefficient__coeff_0__attr__abs__y"] )) +
                0.100000*np.tanh(((data["0__skewness_y"]) + (((data["3__skewness_x"]) + (np.minimum(((((data["4__fft_coefficient__coeff_0__attr__abs__x"]) + (data["flux_median"])))), ((data["0__skewness_y"])))))))) +
                0.100000*np.tanh(((data["flux_d1_pb5"]) + (((data["1__skewness_x"]) + (((((((data["1__kurtosis_x"]) + (data["1__skewness_x"]))) + (data["5__skewness_x"]))) - (data["hostgal_photoz_err"]))))))) +
                0.100000*np.tanh(((-1.0) + (((data["hostgal_photoz"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["hostgal_photoz"]))))))) +
                0.100000*np.tanh(((((data["detected_flux_skew"]) + (data["4__skewness_y"]))) - (data["hostgal_photoz_err"]))) +
                0.100000*np.tanh(((((data["flux_err_min"]) - (data["flux_err_min"]))) + (((data["flux_err_mean"]) * (data["1__skewness_y"]))))) +
                0.100000*np.tanh(((((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (np.where(data["5__skewness_x"]>0, data["detected_flux_max"], data["flux_d1_pb5"] )))) / 2.0)) + (data["1__kurtosis_x"]))) +
                0.100000*np.tanh(np.where(data["detected_flux_err_std"]<0, data["flux_d0_pb5"], data["flux_d0_pb5"] )) +
                0.100000*np.tanh(np.minimum(((((data["3__kurtosis_y"]) - (np.minimum(((data["0__skewness_y"])), ((3.0))))))), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))) +
                0.100000*np.tanh(((np.where(data["hostgal_photoz"] > -1, (-1.0*((data["4__kurtosis_x"]))), data["distmod"] )) + (data["distmod"]))) +
                0.100000*np.tanh(((((((data["hostgal_photoz"]) + (data["hostgal_photoz_err"]))) + (((data["hostgal_photoz"]) + (data["hostgal_photoz_err"]))))) + (data["hostgal_photoz"]))) +
                0.100000*np.tanh(np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__x"])), ((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["detected_mjd_diff"]))))))))



















positives = ['mjd_size',
             'flux_std',
             'flux_err_min',
             'flux_err_max',
             'flux_err_mean',
             'flux_err_median',
             'flux_err_std',
             'flux_ratio_sq_sum',
             'mjd_diff',
             'flux_diff',
             'distmod',
             'detected_mjd_size',
             'detected_flux_std',
             'detected_flux_err_min',
             'detected_flux_err_max',
             'detected_flux_err_mean',
             'detected_flux_err_median',
             'detected_flux_err_std',
             'detected_flux_ratio_sq_sum',
             'detected_mjd_diff',
             'detected_flux_diff',
             '0__fft_coefficient__coeff_0__attr__abs__x',
             '0__fft_coefficient__coeff_1__attr__abs__x',
             '1__fft_coefficient__coeff_0__attr__abs__x',
             '1__fft_coefficient__coeff_1__attr__abs__x',
             '2__fft_coefficient__coeff_0__attr__abs__x',
             '2__fft_coefficient__coeff_1__attr__abs__x',
             '3__fft_coefficient__coeff_0__attr__abs__x',
             '3__fft_coefficient__coeff_1__attr__abs__x',
             '4__fft_coefficient__coeff_0__attr__abs__x',
             '4__fft_coefficient__coeff_1__attr__abs__x',
             '5__fft_coefficient__coeff_0__attr__abs__x',
             '5__fft_coefficient__coeff_1__attr__abs__x',
             '0__fft_coefficient__coeff_0__attr__abs__y',
             '0__fft_coefficient__coeff_1__attr__abs__y',
             '1__fft_coefficient__coeff_0__attr__abs__y',
             '1__fft_coefficient__coeff_1__attr__abs__y',
             '2__fft_coefficient__coeff_0__attr__abs__y',
             '2__fft_coefficient__coeff_1__attr__abs__y',
             '3__fft_coefficient__coeff_0__attr__abs__y',
             '3__fft_coefficient__coeff_1__attr__abs__y',
             '4__fft_coefficient__coeff_0__attr__abs__y',
             '4__fft_coefficient__coeff_1__attr__abs__y',
             '5__fft_coefficient__coeff_0__attr__abs__y',
             '5__fft_coefficient__coeff_1__attr__abs__y']

negatives = ['flux_min',
             'flux_max',
             'flux_mean',
             'flux_median',
             'flux_skew',
             'flux_err_skew',
             'flux_ratio_sq_skew',
             'flux_by_flux_ratio_sq_sum',
             'flux_by_flux_ratio_sq_skew',
             'flux_dif2',
             'flux_w_mean',
             'flux_dif3',
             'flux_d0_pb0',
             'flux_d0_pb1',
             'flux_d0_pb2',
             'flux_d0_pb3',
             'flux_d0_pb4',
             'flux_d0_pb5',
             'flux_d1_pb0',
             'flux_d1_pb1',
             'flux_d1_pb2',
             'flux_d1_pb3',
             'flux_d1_pb4',
             'flux_d1_pb5',
             'detected_flux_min',
             'detected_flux_max',
             'detected_flux_mean',
             'detected_flux_median',
             'detected_flux_skew',
             'detected_flux_err_skew',
             'detected_flux_ratio_sq_skew',
             'detected_flux_by_flux_ratio_sq_sum',
             'detected_flux_by_flux_ratio_sq_skew',
             'detected_flux_dif2',
             'detected_flux_w_mean',
             'detected_flux_dif3',
             '0__kurtosis_x',
             '0__skewness_x',
             '1__kurtosis_x',
             '1__skewness_x',
             '2__kurtosis_x',
             '2__skewness_x',
             '3__kurtosis_x',
             '3__skewness_x',
             '4__kurtosis_x',
             '4__skewness_x',
             '5__kurtosis_x',
             '5__skewness_x',
             '0__kurtosis_y',
             '0__skewness_y',
             '1__kurtosis_y',
             '1__skewness_y',
             '2__kurtosis_y',
             '2__skewness_y',
             '3__kurtosis_y',
             '3__skewness_y',
             '4__kurtosis_y',
             '4__skewness_y',
             '5__kurtosis_y',
             '5__skewness_y']

scl = np.array([0.15892263, 1.05611992, 1.14414584, 1.91505821, 1.7058212 ,
               1.1364961 , 0.71145932, 0.53263496, 0.55325191, 0.5022883 ,
               0.4824709 , 0.49927963, 0.19422397, 0.20310994, 1.72967815,
               0.44219644, 6.77941249, 0.90720406, 0.09639969, 1.06417036,
               2.56774628, 2.50904491, 1.22373783, 0.09954743, 0.44943618,
               0.27531692, 0.07989354, 0.15513414, 1.91763247, 2.06764747,
               2.02777733, 2.01812674, 2.03602114, 2.23189516, 1.65118537,
               2.58910253, 2.58093812, 2.68394528, 2.8051591 , 2.3604408 ,
               0.94356934, 3.09352509, 1.79307058, 2.36116313, 2.52348216,
               1.58296508, 0.53760198, 0.56733886, 0.98992029, 0.73428089,
               0.70716281, 0.87187147, 0.56263831, 2.08539376, 0.56200755,
               6.73568975, 0.63037474, 1.57414742, 1.86198176, 0.9504433 ,
               2.52301198, 0.80726181, 1.49505277, 1.20260641, 0.87036759,
               0.57312047, 1.9701173 , 1.78222973, 1.1233326 , 0.69487555,
               1.86308602, 1.52323012, 1.09224244, 0.6190985 , 1.73787414,
               1.37387029, 1.0693805 , 0.61755715, 1.64168842, 1.25984412,
               0.96996546, 0.60081695, 1.53245533, 1.13973971, 0.80769486,
               0.5254372 , 1.90473482, 1.4674597 , 0.23113522, 0.15510913,
               2.74736345, 2.34244221, 0.33132177, 0.24719373, 2.56340905,
               2.5503959 , 0.50591159, 0.39652209, 2.73349539, 2.50957258,
               0.44483806, 0.3520794 , 3.10806018, 2.50001248, 0.36556932,
               0.27497205, 2.90735646, 2.26488032, 0.26581643, 0.17404689])

mn = np.array([ 4.86129594e+00, -4.44120560e+00,  4.86708140e+00,  1.42428417e+00,
                5.96862723e-01,  3.51477931e+00,  5.02800809e-01,  1.09402228e+00,
                4.11505149e+00,  2.76306865e+00,  2.48206772e+00,  2.58837656e+00,
                8.86992280e-01,  9.83249630e-02,  7.05566452e+00,  1.96547427e+00,
                9.00795496e+00,  1.81908091e+00,  6.87554281e+00,  5.43487482e+00,
                2.94960885e+00,  3.22304030e+00,  1.51535286e+00,  1.00098893e-02,
                5.16459627e-01,  1.53525984e-01,  3.73905154e+00,  9.19439912e-02,
                2.99536517e-01,  7.93390728e-01,  1.34041205e+00,  1.49104722e+00,
                1.39982163e+00,  1.18470004e+00,  2.49966286e-01,  1.27627419e+00,
                2.61969588e+00,  2.52154178e+00,  1.76366663e+00,  6.51390557e-01,
                1.90391821e+00,  2.18818527e+00,  4.28781204e+00,  3.36764546e+00,
                3.25640783e+00,  3.11582075e+00,  5.48038726e-02,  1.48121136e+00,
                2.44474573e+00,  1.99928651e+00,  1.91371374e+00,  1.32571350e+00,
                3.56527672e-01,  6.52941586e+00,  4.30454271e-01,  8.78858941e+00,
                4.04447459e-01,  3.55382590e+00,  3.86104115e+00,  5.17812889e-01,
                3.47054229e+00,  4.58520073e-01,  4.53149838e+00,  3.81385897e+00,
                2.58570639e-01,  1.24608242e-01,  3.92716044e+00,  3.39223920e+00,
                8.28624288e-01,  5.40830021e-01,  5.09724593e+00,  4.62281670e+00,
                1.32386460e+00,  8.51385317e-01,  5.29059262e+00,  4.83466638e+00,
                1.06898538e+00,  7.16690294e-01,  5.86595129e+00,  5.18910662e+00,
                7.98541591e-01,  4.76582520e-01,  6.45403902e+00,  5.51796226e+00,
                5.01700315e-01,  1.88243750e-01,  6.17900289e-01,  3.55486465e-01,
               -5.64772679e-03,  1.39900173e-02,  2.05332554e+00,  1.00812981e+00,
               -6.87464977e-03,  2.92272230e-02,  3.82716288e+00,  2.12225260e+00,
               -2.84775100e-02,  3.18590495e-02,  3.58472311e+00,  1.86610336e+00,
               -3.36801044e-02,  1.05794180e-02,  2.68216171e+00,  1.39686945e+00,
               -3.76596978e-02,  4.01488267e-03,  1.35622789e+00,  7.93381536e-01,
               -3.63165274e-02,  6.68909209e-03])

features = ['mjd_size', 'flux_min', 'flux_max', 'flux_mean', 'flux_median',
           'flux_std', 'flux_skew', 'flux_err_min', 'flux_err_max',
           'flux_err_mean', 'flux_err_median', 'flux_err_std',
           'flux_err_skew', 'detected_mean', 'flux_ratio_sq_sum',
           'flux_ratio_sq_skew', 'flux_by_flux_ratio_sq_sum',
           'flux_by_flux_ratio_sq_skew', 'mjd_diff', 'flux_diff', 'flux_dif2',
           'flux_w_mean', 'flux_dif3', 'ddf', 'hostgal_photoz',
           'hostgal_photoz_err', 'distmod', 'mwebv', 'flux_d0_pb0',
           'flux_d0_pb1', 'flux_d0_pb2', 'flux_d0_pb3', 'flux_d0_pb4',
           'flux_d0_pb5', 'flux_d1_pb0', 'flux_d1_pb1', 'flux_d1_pb2',
           'flux_d1_pb3', 'flux_d1_pb4', 'flux_d1_pb5', 'detected_mjd_size',
           'detected_flux_min', 'detected_flux_max', 'detected_flux_mean',
           'detected_flux_median', 'detected_flux_std', 'detected_flux_skew',
           'detected_flux_err_min', 'detected_flux_err_max',
           'detected_flux_err_mean', 'detected_flux_err_median',
           'detected_flux_err_std', 'detected_flux_err_skew',
           'detected_flux_ratio_sq_sum', 'detected_flux_ratio_sq_skew',
           'detected_flux_by_flux_ratio_sq_sum',
           'detected_flux_by_flux_ratio_sq_skew', 'detected_mjd_diff',
           'detected_flux_diff', 'detected_flux_dif2', 'detected_flux_w_mean',
           'detected_flux_dif3', '0__fft_coefficient__coeff_0__attr__abs__x',
           '0__fft_coefficient__coeff_1__attr__abs__x', '0__kurtosis_x',
           '0__skewness_x', '1__fft_coefficient__coeff_0__attr__abs__x',
           '1__fft_coefficient__coeff_1__attr__abs__x', '1__kurtosis_x',
           '1__skewness_x', '2__fft_coefficient__coeff_0__attr__abs__x',
           '2__fft_coefficient__coeff_1__attr__abs__x', '2__kurtosis_x',
           '2__skewness_x', '3__fft_coefficient__coeff_0__attr__abs__x',
           '3__fft_coefficient__coeff_1__attr__abs__x', '3__kurtosis_x',
           '3__skewness_x', '4__fft_coefficient__coeff_0__attr__abs__x',
           '4__fft_coefficient__coeff_1__attr__abs__x', '4__kurtosis_x',
           '4__skewness_x', '5__fft_coefficient__coeff_0__attr__abs__x',
           '5__fft_coefficient__coeff_1__attr__abs__x', '5__kurtosis_x',
           '5__skewness_x', '0__fft_coefficient__coeff_0__attr__abs__y',
           '0__fft_coefficient__coeff_1__attr__abs__y', '0__kurtosis_y',
           '0__skewness_y', '1__fft_coefficient__coeff_0__attr__abs__y',
           '1__fft_coefficient__coeff_1__attr__abs__y', '1__kurtosis_y',
           '1__skewness_y', '2__fft_coefficient__coeff_0__attr__abs__y',
           '2__fft_coefficient__coeff_1__attr__abs__y', '2__kurtosis_y',
           '2__skewness_y', '3__fft_coefficient__coeff_0__attr__abs__y',
           '3__fft_coefficient__coeff_1__attr__abs__y', '3__kurtosis_y',
           '3__skewness_y', '4__fft_coefficient__coeff_0__attr__abs__y',
           '4__fft_coefficient__coeff_1__attr__abs__y', '4__kurtosis_y',
           '4__skewness_y', '5__fft_coefficient__coeff_0__attr__abs__y',
           '5__fft_coefficient__coeff_1__attr__abs__y', '5__kurtosis_y',
           '5__skewness_y']

def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion.png')

def photoztodist(data) :
    return ((((((np.log((((data["hostgal_photoz"]) + (np.log((((data["hostgal_photoz"]) + (np.sqrt((np.log((np.maximum(((3.0)), ((((data["hostgal_photoz"]) * 2.0))))))))))))))))) + ((12.99870681762695312)))) + ((1.17613816261291504)))) * (3.0))

fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None}

def get_inputs(data,metadata,slices=20):
    agg_df_ts = None
    agg_df_ts_detected = None
    for i in range(slices):
        uniqueids = data.object_id.unique()[i::slices]
        sub = data[data.object_id.isin(uniqueids)].copy()
        x = sub.groupby(['object_id','passband'])['mjd','flux'].diff().fillna(0)
        x['object_id'] = sub.object_id
        x['passband'] = sub.passband
        x = x.groupby(['object_id','passband'])['mjd','flux'].cumsum().fillna(0)
        x['object_id'] = sub.object_id
        x['passband'] = sub.passband
        x['detected'] = sub.detected
        df_ts = extract_features(x, column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4)
        df_ts.index.rename('object_id',inplace=True)
        df_ts.reset_index(drop=False,inplace=True)
        df_ts_detected = extract_features(x.loc[(x.detected==1)], column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4)
        df_ts_detected.index.rename('object_id',inplace=True)
        df_ts_detected.reset_index(drop=False,inplace=True)
        if(agg_df_ts is None):
            agg_df_ts = df_ts.copy()
            agg_df_ts_detected = df_ts_detected.copy()
        else:
            agg_df_ts = pd.concat([agg_df_ts,df_ts.fillna(0)],sort=False)
            agg_df_ts_detected = pd.concat([agg_df_ts_detected,df_ts_detected.fillna(0)],sort=False)
        del df_ts, df_ts_detected, x, sub
        gc.collect()
    
    for d in [0,1]:
        for pb in range(6):
            x = None
            if(d==0):
                x = data[(data.passband==pb)][['object_id','flux']].groupby(['object_id']).flux.mean().reset_index(drop=False)
            else:
                x = data[(data.passband==pb)&(data.detected==1)][['object_id','flux']].groupby(['object_id']).flux.mean().reset_index(drop=False)
            x.columns = ['object_id','flux_d'+str(d)+'_pb'+str(pb)]  
            metadata = metadata.merge(x,on='object_id',how='left')
            del x
            gc.collect()
    
    data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
    data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }
    x = data[data.detected==1].groupby(['object_id']).agg(aggs)
    new_columns = [
            k + '_' + agg for k in aggs.keys() for agg in aggs[k]
        ]
    x.columns = new_columns

    x['mjd_diff'] = x['mjd_max'] - x['mjd_min']
    x['flux_diff'] = x['flux_max'] - x['flux_min']
    x['flux_dif2'] = (x['flux_max'] - x['flux_min']) / x['flux_mean']
    x['flux_w_mean'] = x['flux_by_flux_ratio_sq_sum'] / x['flux_ratio_sq_sum']
    x['flux_dif3'] = (x['flux_max'] - x['flux_min']) / x['flux_w_mean']
    del x['mjd_max'], x['mjd_min']
    x.columns = ['detected_'+c for c in x.columns]
    x = x.reset_index(drop=False)
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_data = data.groupby(['object_id']).agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_data.columns = new_columns
    agg_data['mjd_diff'] = agg_data['mjd_max'] - agg_data['mjd_min']
    agg_data['flux_diff'] = agg_data['flux_max'] - agg_data['flux_min']
    agg_data['flux_dif2'] = (agg_data['flux_max'] - agg_data['flux_min']) / agg_data['flux_mean']
    agg_data['flux_w_mean'] = agg_data['flux_by_flux_ratio_sq_sum'] / agg_data['flux_ratio_sq_sum']
    agg_data['flux_dif3'] = (agg_data['flux_max'] - agg_data['flux_min']) / agg_data['flux_w_mean']
    del agg_data['mjd_max'], agg_data['mjd_min']
    agg_data.reset_index(drop=False,inplace=True)
    full_data = agg_data.merge(right=metadata, how='outer', on=['object_id'])
    del full_data['hostgal_specz']
    del full_data['ra'], full_data['decl'], full_data['gal_l'],full_data['gal_b']
    del data, metadata
    gc.collect()
    full_data = full_data.merge(x,on=['object_id'],how='left')
    full_data = full_data.merge(agg_df_ts,on=['object_id'],how='left')
    full_data = full_data.merge(agg_df_ts_detected,on=['object_id'],how='left')
    full_data.loc[~(full_data.hostgal_photoz>0),'distmod'] = photoztodist(full_data.loc[~(full_data.hostgal_photoz>0)])
    full_data.columns = [c.replace('"','_')for c in full_data.columns]
    for i, c in enumerate(features):
        if(c in negatives):
            full_data.loc[~full_data[c].isnull(),c] = np.sign(full_data.loc[~full_data[c].isnull(),c])*np.log1p(np.abs(full_data.loc[~full_data[c].isnull(),c]))
        elif(c in positives):
            full_data.loc[~full_data[c].isnull(),c] = np.log1p(full_data.loc[~full_data[c].isnull(),c])

    full_data.fillna(0,inplace=True)
    ss = StandardScaler()
    ss.scale_ = scl
    ss.mean_ = mn
    full_data.loc[:,features] = ss.transform(full_data.loc[:,features])
    return full_data


def GenerateConfusionMatrix():
    gpI = GPSoftmax()

    meta_train = pd.read_csv('../input/training_set_metadata.csv')
    train = pd.read_csv('../input/training_set.csv')
    full_train = get_inputs(train,meta_train,20)
    del meta_train, train
    gc.collect()
    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']
    classes = sorted(y.unique())

    class_weight = {
        c: 1 for c in classes
    }
    for c in [64, 15]:
        class_weight[c] = 2

    print('Unique classes : ', classes)

    unique_y = np.unique(y)
    class_map = dict()
    for i,val in enumerate(unique_y):
        class_map[val] = i
            
    y_map = np.zeros((y.shape[0],))
    y_map = np.array([class_map[val] for val in y])
    y_categorical = to_categorical(y_map)

    sample_sub = pd.read_csv('../input/sample_submission.csv')
    class_names = list(sample_sub.columns[1:-1])
    del sample_sub;gc.collect()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
    oof_preds = np.zeros((len(full_train), len(classes)))
    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
        x_train, y_train = full_train.iloc[trn_], y_categorical[trn_]
        x_valid, y_valid = full_train.iloc[val_], y_categorical[val_]
        # Get predicted probabilities for each class
        valpreds = gpI.GrabPredictions(x_valid).values
        print(fold_)
        print('LOG LOSS {0}: {1}'.format(fold_ , log_loss(y_valid,valpreds)))    
        print('MULTI WEIGHTED LOG LOSS {0}: {1}'.format(fold_ , multi_weighted_logloss(y_valid,valpreds)))
        oof_preds[val_, :] = valpreds
    print('Final LOG LOSS : %.5f ' % log_loss(y_categorical,oof_preds))    
    print('Final MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_categorical,oof_preds))
    cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
    np.set_printoptions(precision=2)
    plt.figure(figsize=(12,12))
    foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                          title='Confusion Matix') 


if __name__=="__main__":
    GenerateConfusionMatrix()