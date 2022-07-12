import os
import re
from glob import glob
from time import time, strftime, gmtime
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib
import matplotlib.pyplot as plt


TRAIN_TFREC_FORMAT = {
    'recording_id': tf.io.FixedLenFeature([], tf.string),
    'label_info': tf.io.FixedLenFeature([], tf.string),
    'audio_wav': tf.io.FixedLenFeature([], tf.string)
}
TEST_TFREC_FORMAT = {
    'recording_id': tf.io.FixedLenFeature([], tf.string),
    'audio_wav': tf.io.FixedLenFeature([], tf.string)
}
NORM_ENABLE = True
PLOT_ENABLE = False

# All classes have a number of labelled samples between 300 and 350: the following parameters are chosen accordingly
# (CV_FOLDS_NUM*CV_FOLD_SIZE should be less than the minimum size of the dataset for each class)
SHUFFLE_BUFFER_SIZE = 350
SHUFFLE_SEED = 42
CV_FOLDS_NUM = 3
CV_FOLD_SIZE = 100

AUTOTUNE = tf.data.experimental.AUTOTUNE
SR = 48000
F_MIN = 0
F_MAX = SR/2

# Possible (species_id, songtype_id) combinations are 26, given by all 24 values of species_id plus 2 more entries for
# the species ID having more than a single songtype_id
SPECIES_SONGTYPE_TO_CATEGORY_DICT = {
    (0,  1): 0,
    (1,  1): 1,
    (2,  1): 2,
    (3,  1): 3,
    (4,  1): 4,
    (5,  1): 5,
    (6,  1): 6,
    (7,  1): 7,
    (8,  1): 8,
    (9,  1): 9,
    (10, 1): 10,
    (11, 1): 11,
    (12, 1): 12,
    (13, 1): 13,
    (14, 1): 14,
    (15, 1): 15,
    (16, 4): 16,
    (17, 1): 17,
    (18, 1): 18,
    (19, 1): 19,
    (20, 1): 20,
    (21, 1): 21,
    (22, 1): 22,
    (23, 1): 23,
    (17, 4): 24,
    (23, 4): 25
}


# Define a custom Keras layer for the calculation of Log Mel spectrogram, with parameters
# Credits to David Schwertfeger:
# https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                 f_min=0.0, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.

        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.

        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def power_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = power_to_db(mel_spectrograms)

        # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config


def tf_parse_train(raw_data):
    example = tf.io.parse_example(raw_data[tf.newaxis], TRAIN_TFREC_FORMAT)
    recording_id = example['recording_id'][0]
    label_info = example['label_info'][0]
    audio_wav = example['audio_wav'][0]

    wav_data, _ = tf.audio.decode_wav(audio_wav)
    return recording_id, wav_data, label_info


def tf_build_label(recording_id, wav_data, label_info):
    # Split label_info into individual annotations for the single recording (discard time and freq localization)
    label_list = re.findall('\d+,\d+,\d+.\d+,\d+.\d+,\d+.\d+,\d+.\d+,[0-1]', label_info.numpy().decode('utf-8'))
    species_id = list()
    songtype_id = list()
    is_tp = list()
    for item in label_list:
        item_species_id, item_songtype_id, _, _, _, _, item_is_tp = item.split(',')
        species_id.append(item_species_id)
        songtype_id.append(item_songtype_id)
        is_tp.append(item_is_tp)

    # Assign label_data as an array of ground truth values for all (species_id, songtype_id) combinations
    # (keep value 0.5 for all categories which have been labelled as no TP/FP or both TP/FP for the current recording)
    label_data = np.ones(len(SPECIES_SONGTYPE_TO_CATEGORY_DICT))*0.5
    # print('Recording ID: {}, Species ID: {}, Songtype ID: {}, is_tp: {}'.format(
    #     recording_id, species_id, songtype_id, is_tp))
    for idx in range(len(is_tp)):
        category_idx = SPECIES_SONGTYPE_TO_CATEGORY_DICT[int(species_id[idx]), int(songtype_id[idx])]
        if label_data[category_idx] == 0.5:
            label_data[category_idx] = 1.0 if int(is_tp[idx]) else 0.0
        elif (label_data[category_idx] == 0.0 and int(is_tp[idx])) or \
             (label_data[category_idx] == 1.0 and not(int(is_tp[idx]))):
            label_data[category_idx] = 0.5
    # print('Label data: {}'.format(label_data))

    return recording_id, wav_data, label_data


def tf_parse_test(raw_data):
    example = tf.io.parse_example(raw_data[tf.newaxis], TEST_TFREC_FORMAT)
    recording_id = example['recording_id'][0]
    audio_wav = example['audio_wav'][0]

    wav_data, _ = tf.audio.decode_wav(audio_wav)
    return recording_id, wav_data


def create_data_folds(train_tfrec_dir, target_species_songtype):
    # Prepare overall training data
    train_files = glob(train_tfrec_dir + '/*.tfrec')
    train_data = tf.data.TFRecordDataset(filenames=train_files)
    train_data = train_data.map(tf_parse_train, num_parallel_calls=AUTOTUNE)
    train_data = train_data.map(lambda a, b, c: tf.py_function(func=tf_build_label, inp=[a, b, c],
                                                               Tout=[tf.string, tf.float32, tf.float32]),
                                num_parallel_calls=AUTOTUNE)
    train_data = train_data.filter(lambda a, b, c: c[SPECIES_SONGTYPE_TO_CATEGORY_DICT[target_species_songtype]] != 0.5)
    train_data = train_data.map(lambda a, b, c: (a, b, c[SPECIES_SONGTYPE_TO_CATEGORY_DICT[target_species_songtype]]))
    train_data = train_data.map(lambda a, b, c: (tf.ensure_shape(b, [60*SR, 1]), tf.ensure_shape(c, [])))
    train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE, seed=SHUFFLE_SEED, reshuffle_each_iteration=False)
    train_data = train_data.cache()

    # Create CV_FOLDS_NUM folds of data, plus a last one with the remainder of records given by the integer division of
    # the overall samples number by CV_FOLD_SIZE
    train_data_folds = list()
    for fold_idx in range(CV_FOLDS_NUM):
        train_data_folds.append(train_data.skip(fold_idx*CV_FOLD_SIZE).take(CV_FOLD_SIZE))
    train_data_folds.append(train_data.skip(CV_FOLDS_NUM * CV_FOLD_SIZE))

    return train_data_folds


def create_model(n_hop, n_mels, n_filters, dropout_rate, l2_reg_rate):
    # Create neural network layers
    # The output of LogMelSpectrogram has a size given by: H ~ 60*SR/n_hop, W = n_mels, C = 1
    model_input = layers.Input(shape=60 * SR)
    model_mel = LogMelSpectrogram(SR, n_hop*4, n_hop, n_mels, f_min=F_MIN, f_max=F_MAX)
    model_norm = preprocessing.Normalization()
    model_reshape = layers.Reshape((-1, n_mels))
    model_conv1d_0 = layers.Conv1D(n_filters, 3, kernel_initializer='he_uniform', activation='relu')
    model_bnorm_0 = layers.BatchNormalization()
    model_dropout_0 = layers.Dropout(dropout_rate)
    model_pool1d_0 = layers.MaxPooling1D()
    model_flatten_0 = layers.Flatten()
    model_conv1d_1 = layers.Conv1D(n_filters, 5, kernel_initializer='he_uniform', activation='relu')
    model_bnorm_1 = layers.BatchNormalization()
    model_dropout_1 = layers.Dropout(dropout_rate)
    model_pool1d_1 = layers.MaxPooling1D()
    model_flatten_1 = layers.Flatten()
    model_conv1d_2 = layers.Conv1D(n_filters, 11, kernel_initializer='he_uniform', activation='relu')
    model_bnorm_2 = layers.BatchNormalization()
    model_dropout_2 = layers.Dropout(dropout_rate)
    model_pool1d_2 = layers.MaxPooling1D()
    model_flatten_2 = layers.Flatten()
    model_concat = layers.Concatenate()
    model_dense_0 = layers.Dense(n_filters, activation='relu')
    model_bnorm_3 = layers.BatchNormalization()
    model_dense_1 = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=l2_reg_rate))

    # Create neural network connections
    model_mel_o = model_mel(model_input)
    model_norm_o = model_norm(model_mel_o)
    model_reshape_o = model_reshape(model_norm_o)
    model_conv1d_0_o = model_conv1d_0(model_reshape_o)
    model_bnorm_0_o = model_bnorm_0(model_conv1d_0_o)
    model_dropout_0_o = model_dropout_0(model_bnorm_0_o)
    model_pool1d_0_o = model_pool1d_0(model_dropout_0_o)
    model_flatten_0_o = model_flatten_0(model_pool1d_0_o)
    model_conv1d_1_o = model_conv1d_1(model_reshape_o)
    model_bnorm_1_o = model_bnorm_1(model_conv1d_1_o)
    model_dropout_1_o = model_dropout_1(model_bnorm_1_o)
    model_pool1d_1_o = model_pool1d_1(model_dropout_1_o)
    model_flatten_1_o = model_flatten_1(model_pool1d_1_o)
    model_conv1d_2_o = model_conv1d_2(model_reshape_o)
    model_bnorm_2_o = model_bnorm_2(model_conv1d_2_o)
    model_dropout_2_o = model_dropout_2(model_bnorm_2_o)
    model_pool1d_2_o = model_pool1d_2(model_dropout_2_o)
    model_flatten_2_o = model_flatten_2(model_pool1d_2_o)
    model_concat_o = model_concat([model_flatten_0_o, model_flatten_1_o, model_flatten_2_o])
    model_dense_0_o = model_dense_0(model_concat_o)
    model_bnorm_3_o = model_bnorm_3(model_dense_0_o)
    model_dense_1_o = model_dense_1(model_bnorm_3_o)

    # Create and summarize end-to-end neural network model
    result = tf.keras.Model(inputs=model_input, outputs=model_dense_1_o)

    return result


def train_model(target_species_songtype,
                n_mels=256,
                n_hop=2048,
                n_filters=16,
                learn_rate=0.0005,
                l2_reg_rate=0.05,
                dropout_rate=0.5,
                batch_size=4,
                epochs_num=10):

    # Define all needed paths
    param_str = 'NM{}_NH{}_NF{}_BS{}_LR{}_DR{}_RR{}'.format(n_mels, n_hop, n_filters, batch_size, learn_rate,
                                                            dropout_rate, l2_reg_rate)
    target_str = 'target_{}_{}'.format(*target_species_songtype)
    datetime_str = '@' + datetime.now().strftime('%y%m%d%H%M%S')
    logs_dir = os.path.join('_summaries/rfcx-species-audio-detection', param_str, target_str, datetime_str)

    if os.getcwd() == '/kaggle/working':
        train_tfrec_dir = '../input/rfcx-species-audio-detection/tfrecords/train'
        models_dir = os.path.join('models', param_str)
    else:
        train_tfrec_dir = 'data/rfcx-species-audio-detection/tfrecords/train'
        models_dir = os.path.join('projects/rfcx-species-audio-detection/models', param_str)
        matplotlib.use('QT5Agg')

    # Create directories for models as needed
    model_target_dir = os.path.join(models_dir, target_str)
    if not (os.path.exists(model_target_dir)):
        os.makedirs(model_target_dir)

    # Shuffle and split all available training data into two parts, respectively for actual training and for validation
    train_data_folds = create_data_folds(train_tfrec_dir, target_species_songtype)
    
    # Prepare a list for each metric to be saved across CV folds
    folds_val_accuracy = list()
    folds_val_f1_score = list()
    
    train_beg_time = time()
    for fold_idx in range(CV_FOLDS_NUM+1):
        if fold_idx < CV_FOLDS_NUM:
            # Use the fold indexed by fold_idx as the validation fold and the concatenation of all the other ones for
            # training (when concatenating all other folds, trn_ds is initialized to the records remaining in the last
            # fold, which therefore are always considered only as training data)
            val_ds = train_data_folds[fold_idx]
            trn_ds = train_data_folds[-1]
            for idx in range(CV_FOLDS_NUM):
                if idx == fold_idx:
                    continue
                trn_ds = trn_ds.concatenate(train_data_folds[idx])
            val_ds = val_ds.batch(1).prefetch(AUTOTUNE)
            trn_ds = trn_ds.batch(batch_size).prefetch(AUTOTUNE)
        else:
            # At the last iteration, build up a model using all the available data for training: this will be the final
            # model, with the expected validation performance already evaluated through CV
            trn_ds = train_data_folds[-1]
            for idx in range(CV_FOLDS_NUM):
                trn_ds = trn_ds.concatenate(train_data_folds[idx])
            val_ds = None
            trn_ds = trn_ds.batch(batch_size).prefetch(AUTOTUNE)

        # Trn and val dataset are now defined: create the model from scratch and train it
        model = create_model(n_hop, n_mels, n_filters, dropout_rate, l2_reg_rate)
        model.summary()

        norm_layer = model.layers[2]
        if NORM_ENABLE:
            # Evaluate mean and variance of the LogMelSpectrogram output in order to initialize a normalization layer
            k_input = layers.Input(shape=60*SR)
            k_mel = LogMelSpectrogram(SR, n_hop*4, n_hop, n_mels, f_min=F_MIN, f_max=F_MAX)(k_input)
            norm_layer.adapt(tf.keras.Model(inputs=k_input, outputs=k_mel).predict(trn_ds))
        else:
            # Create a "transparent" normalization layer, by initializing mean = 0 and variance = 1
            norm_layer.adapt([-1, 1])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        callbacks = []
        if not(os.getcwd() == '/kaggle/working'):
            if fold_idx < CV_FOLDS_NUM:
                tensorboard_cb = tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(logs_dir, 'fold_{}'.format(fold_idx)))
                callbacks.append(tensorboard_cb)
        if fold_idx < CV_FOLDS_NUM:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_target_dir, 'fold_{}.h5'.format(fold_idx)),
                monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
            callbacks.append(checkpoint_cb)

        history = model.fit(
            trn_ds,
            epochs=epochs_num,
            validation_data=val_ds,
            callbacks=callbacks
        )

        if PLOT_ENABLE:
            metrics = history.history
            plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
            plt.legend(['loss', 'val_loss'])
            plt.show()
            plt.figure()
            plt.plot(history.epoch, metrics['val_accuracy'])
            plt.legend(['accuracy', 'val_accuracy'])
            plt.show()

        # While on cross-validation, store performance results; at the end, save the final model
        if fold_idx < CV_FOLDS_NUM:
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for val_el in val_ds:
                label = val_el[1]
                prob = model(np.transpose(val_el[0]))
                pred = tf.round(prob)[0]
                # print('label = {}, probability = {}, prediction = {}'.format(label, prob, pred))
                tp += 1 if (label and pred) else 0
                fp += 1 if (not label and pred) else 0
                fn += 1 if (label and not pred) else 0
                tn += 1 if (not label and not pred) else 0
            accuracy = (tp+tn)/(tp+fp+fn+tn+np.finfo(float).eps)
            precision = tp/(tp+fp+np.finfo(float).eps)
            recall = tp/(tp+fn+np.finfo(float).eps)
            f1_score = 2*precision*recall/(precision+recall+np.finfo(float).eps)
            
            with open(os.path.join(model_target_dir, "fold_{}.log".format(fold_idx)), 'w') as fid:
                fid.write('Validation metrics for fold {}\n'.format(fold_idx))
                fid.write('|- TP = {:3}, FP = {:3}\n'.format(tp, fp))
                fid.write('|- FN = {:3}, TN = {:3}\n'.format(fn, tn))
                fid.write('|- Accuracy  = {:.3f}\n'.format(accuracy))
                fid.write('|- Precision = {:.3f}\n'.format(precision))
                fid.write('|- Recall    = {:.3f}\n'.format(recall))
                fid.write('|- F1 score  = {:.3f}\n'.format(f1_score))

            folds_val_accuracy.append(accuracy)
            folds_val_f1_score.append(f1_score)
        else:
            mean_val_accuracy = np.mean(np.array(folds_val_accuracy))
            mean_val_f1_score = np.mean(np.array(folds_val_f1_score))
            std_val_accuracy = np.std(np.array(folds_val_accuracy))
            std_val_f1_score = np.std(np.array(folds_val_f1_score))
            with open(os.path.join(model_target_dir, "model.log"), 'w') as fid:
                fid.write('Validation metrics for the final model\n')
                fid.write('|- Accuracy  = {:.3f} (std: {:.3f})\n'.format(mean_val_accuracy, std_val_accuracy))
                fid.write('|- F1 score  = {:.3f} (std: {:.3f})\n'.format(mean_val_f1_score, std_val_f1_score))
            model.save(os.path.join(model_target_dir, "model.h5"))

        del model

    train_end_time = time()
    train_delta_hhmmss = strftime('%H:%M:%S', gmtime(train_end_time-train_beg_time))

    print("Model overall training execution time (hh:mm:ss) = {}".format(train_delta_hhmmss))


def test_model(n_mels=256,
               n_hop=2048,
               n_filters=16,
               learn_rate=0.0005,
               l2_reg_rate=0.05,
               dropout_rate=0.5,
               batch_size=4):
    param_str = 'NM{}_NH{}_NF{}_BS{}_LR{}_DR{}_RR{}'.format(n_mels, n_hop, n_filters, batch_size, learn_rate,
                                                            dropout_rate, l2_reg_rate)

    # Create and summarize the model
    model = create_model(n_hop, n_mels, n_filters, dropout_rate, l2_reg_rate)
    model.summary()

    # Prepare the paths to load the data weights of the model from
    if os.getcwd() == '/kaggle/working':
        test_tfrec_dir = '../input/rfcx-species-audio-detection/tfrecords/test'
        models_dir = os.path.join('models', param_str)
    else:
        test_tfrec_dir = 'data/rfcx-species-audio-detection/tfrecords/test'
        models_dir = os.path.join('projects/rfcx-species-audio-detection/models', param_str)

    # Prepare test-data
    test_files = glob(test_tfrec_dir + '/*.tfrec')
    test_data = tf.data.TFRecordDataset(filenames=test_files)
    test_data = test_data.map(tf_parse_test, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(lambda a, b: (a, tf.ensure_shape(b, [60*SR, 1])))
    test_audio_ds = test_data.map(lambda a, b: tf.transpose(b))

    # Prepare destination data-frame
    test_filenames_ds = test_data.map(lambda a, b: a)
    filenames = [item.numpy().decode() for item in test_filenames_ds]
    df = pd.DataFrame()
    df["recording_id"] = filenames
    df.set_index("recording_id", inplace=True)

    # For each species and songtype, load the model weights, make the predictions and store them in the result dataframe
    for item in SPECIES_SONGTYPE_TO_CATEGORY_DICT:
        print("Making predictions for species/songtype ({},{})".format(*item))
        model.load_weights(os.path.join(models_dir, 'target_{}_{}'.format(*item), 'model.h5'))
        df["sp{}so{}".format(*item)] = model.predict(test_audio_ds)
    df = df.sort_index()

    # Produce the result dataframe and build submission CSV file from it
    result_df = pd.DataFrame()
    result_df["recording_id"] = df.index
    result_df.set_index("recording_id", inplace=True)
    for idx in range(24):
        if idx == 16:
            result_df["s{}".format(idx)] = df["sp{}so4".format(idx)]
        elif idx == 17 or idx == 23:
            result_df["s{}".format(idx)] = df[["sp{}so1".format(idx), "sp{}so4".format(idx)]].max(axis=1)
        else:
            result_df["s{}".format(idx)] = df["sp{}so1".format(idx)]
    result_df.to_csv(os.path.join(models_dir, "submission.csv"))

    return result_df


train_model((0, 1))