# %% [code] {"_kg_hide-input":false}
import os
import re
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt

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


if os.getcwd() == "/kaggle/working":
    TRAIN_TFREC_DIR = "../input/rfcx-species-audio-detection/tfrecords/train"
else:
    TRAIN_TFREC_DIR = "data/rfcx-species-audio-detection/tfrecords/train"
    matplotlib.use("QT5Agg")

TRAIN_TFREC_FORMAT = {
    'recording_id': tf.io.FixedLenFeature([], tf.string),
    'label_info': tf.io.FixedLenFeature([], tf.string),
    'audio_wav': tf.io.FixedLenFeature([], tf.string)
}

# Since the shortest label duration is about 300 ms and SR is 48 KHz, suitable values for N_HOP should 
# be the following ones: 1024 (21 ms), 2048, 4096 (85 ms). Smaller values of N_HOP give neural networks 
# with more weights
N_HOP = 2048
# A bigger value of N_MELS gives greater details in the Mel spectrogram on the frequency axis, but 
# costs more neural network weights, so it should be kept as small as possible. A suitable value may 
# be 128 (or also 256)
N_MELS = 128

AUTOTUNE = tf.data.experimental.AUTOTUNE
SR = 48000
F_MIN = 0
F_MAX = SR/2
N_FFT = N_HOP*4

# Possible (species_id, songtype_id) combinations are 26, given by all 24 values of species_id plus 2 
# more entries for the species IDs having more than a single songtype ID
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
TARGET_SPECIES_SONGTYPE = (3, 1)


def tf_parse(raw_data):
    example = tf.io.parse_example(raw_data[tf.newaxis], TRAIN_TFREC_FORMAT)
    recording_id = example['recording_id'][0]
    label_info = example['label_info'][0]
    audio_wav = example['audio_wav'][0]

    wav_data, _ = tf.audio.decode_wav(audio_wav)
    return recording_id, wav_data, label_info


def tf_build_label(recording_id, wav_data, label_info):
    # Split label_info into individual annotations for the single recording (discard 
    # time and freq localization)
    label_list = re.findall("\d+,\d+,\d+.\d+,\d+.\d+,\d+.\d+,\d+.\d+,[0-1]", label_info.numpy().decode('utf-8'))
    species_id = list()
    songtype_id = list()
    category_id = list()
    t_min = list()
    f_min = list()
    t_max = list()
    f_max = list()
    is_tp = list()
    for item in label_list:
        item_species_id, item_songtype_id, item_t_min, item_f_min, item_t_max, item_f_max, item_is_tp = item.split(",")
        species_id.append(item_species_id)
        songtype_id.append(item_songtype_id)
        category_id.append(SPECIES_SONGTYPE_TO_CATEGORY_DICT[int(item_species_id), int(item_songtype_id)])
        t_min.append(float(item_t_min))
        f_min.append(float(item_f_min))
        t_max.append(float(item_t_max))
        f_max.append(float(item_f_max))
        is_tp.append(int(item_is_tp))

    # Assign label_data as an array of ground truth values for all (species_id, songtype_id) 
    # combinations (keep value 0.5 for all categories which have been labelled as no TP/FP or 
    # both TP/FP for the current recording)
    label_data = np.ones(len(SPECIES_SONGTYPE_TO_CATEGORY_DICT))*0.5
    # print("Recording ID: {}, Species ID: {}, Songtype ID: {}, is_tp: {}".format(
    #     recording_id, species_id, songtype_id, is_tp))
    for idx in range(len(is_tp)):
        category_idx = SPECIES_SONGTYPE_TO_CATEGORY_DICT[int(species_id[idx]), int(songtype_id[idx])]
        if label_data[category_idx] == 0.5:
            label_data[category_idx] = 1.0 if is_tp[idx] else 0.0
        elif (label_data[category_idx] == 0.0 and is_tp[idx]) or \
             (label_data[category_idx] == 1.0 and not(is_tp[idx])):
            label_data[category_idx] = 0.5
    # print("Label data: {}".format(label_data))

    return recording_id, tf.ensure_shape(wav_data, [60 * SR, 1]), label_data, category_id, t_min, f_min, t_max, f_max


# Prepare training data
train_files = glob(TRAIN_TFREC_DIR + "/*.tfrec")
train_data = tf.data.TFRecordDataset(filenames=train_files)
train_data = train_data.map(tf_parse, num_parallel_calls=AUTOTUNE)
train_data = train_data.map(lambda a, b, c: tf.py_function(func=tf_build_label, inp=[a, b, c],
                                                           Tout=[tf.string, tf.float32, tf.float32, tf.int8,
                                                                 tf.float32, tf.float32, tf.float32, tf.float32]),
                            num_parallel_calls=AUTOTUNE)
train_data = train_data.filter(lambda a, b, c, d, e, f, g, h:
                               c[SPECIES_SONGTYPE_TO_CATEGORY_DICT[TARGET_SPECIES_SONGTYPE]] != 0.5)

# Prepare model to evaluate LogMelSpectrogram
k_input = layers.Input(shape=60 * SR)
k_mel = LogMelSpectrogram(SR, N_FFT, N_HOP, N_MELS, f_min=F_MIN, f_max=F_MAX)
mel_model = tf.keras.Model(inputs=k_input, outputs=k_mel(k_input))
ftm_mtx = k_mel.mel_filterbank


# Extract label data for configured TARGET_SPECIES_SONGTYPE (in case of more time and 
# frequency labels, take the first one): each time this function is called, a plot of 
# the next labelled data will be produced
def plot_label_on_mel(data_item):
    mel_item = mel_model.predict(np.transpose(data_item[1].numpy()))
    is_tp = data_item[2].numpy()[SPECIES_SONGTYPE_TO_CATEGORY_DICT[TARGET_SPECIES_SONGTYPE]]
    category_id = data_item[3].numpy()
    t_min = data_item[4].numpy()[np.nonzero(category_id == SPECIES_SONGTYPE_TO_CATEGORY_DICT[TARGET_SPECIES_SONGTYPE])][0]
    f_min = data_item[5].numpy()[np.nonzero(category_id == SPECIES_SONGTYPE_TO_CATEGORY_DICT[TARGET_SPECIES_SONGTYPE])][0]
    t_max = data_item[6].numpy()[np.nonzero(category_id == SPECIES_SONGTYPE_TO_CATEGORY_DICT[TARGET_SPECIES_SONGTYPE])][0]
    f_max = data_item[7].numpy()[np.nonzero(category_id == SPECIES_SONGTYPE_TO_CATEGORY_DICT[TARGET_SPECIES_SONGTYPE])][0]
    print("Label localization: t=({:0.2f},{:0.2f}), f=({:.2f},{:.2f})".format(t_min, t_max, f_min, f_max))
    n_min = t_min/60*mel_item.shape[1]
    n_max = t_max/60*mel_item.shape[1]
    m_min = np.max(np.nonzero(ftm_mtx[int(np.round((f_min-F_MIN)/(F_MAX-F_MIN)*(ftm_mtx.shape[0]-1)))]), initial=0)
    m_max = np.min(np.nonzero(ftm_mtx[int(np.round((f_max-F_MIN)/(F_MAX-F_MIN)*(ftm_mtx.shape[0]-1)))]), initial=N_MELS)

    # Plot the time signal and its Mel spectrogram and highlight the label t/f bbox 
    # in black for TP and in red for FP
    bbox_color = 'k' if is_tp else 'r'
    fig, axes = plt.subplots(2, figsize=(12, 8))
    axes[0].set_title("Audio signal (ID: {})".format(data_item[0].numpy().decode()))
    axes[0].plot(np.arange(0, 60, 1/SR), data_item[1].numpy().squeeze())
    axes[0].set_xlim([0, 60])
    axes[0].plot([t_min, t_min], [-1, 1], color=bbox_color, linestyle='-', linewidth=2)
    axes[0].plot([t_max, t_max], [-1, 1], color=bbox_color, linestyle='-', linewidth=2)
    axes[1].set_title("Mel spectrogram (N_HOP={}, N_MELS={})".format(N_HOP, N_MELS))
    axes[1].pcolormesh(mel_item.squeeze().transpose())
    axes[1].plot([n_min, n_min], [m_min, m_max], color=bbox_color, linestyle='-', linewidth=2)
    axes[1].plot([n_min, n_max], [m_min, m_min], color=bbox_color, linestyle='-', linewidth=2)
    axes[1].plot([n_min, n_max], [m_max, m_max], color=bbox_color, linestyle='-', linewidth=2)
    axes[1].plot([n_max, n_max], [m_min, m_max], color=bbox_color, linestyle='-', linewidth=2)


# Create an iterator of the data: example Mel spectrogram plots for the given 
# TARGET_SPECIES_SONGTYPE can be produced by calling the plot_label_on_mel on 
# the next iterator element
data_iter = iter(train_data)
plot_label_on_mel(next(data_iter))
