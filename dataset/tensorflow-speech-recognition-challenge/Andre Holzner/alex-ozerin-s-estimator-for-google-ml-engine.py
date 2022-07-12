#!/usr/bin/env python
# 
# This is Alex Ozerin's notebook kernel (https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72)
# adapted to use with Google mlengine. 
# 
# Changes are mainly related to accessing files, making 
# some parameters configurable from the command line to 
# take into account differences of local running (testing)
# and running on the mlengine (reading data from a storage bucket)
# and logging.
# Also made some other parameters configurable from the command
# line (such as number of iterations) for quick testing.
#
# Note that mlengine currently does not support python 3
#
# ----------
# To run this:
#
#  * create a directory 'trainer' (without quotes)
#  * save this file as trainer/task.py
#  * create an empty file trainer/__init__.py:
#      touch trainer/__init__.py
#
# To run this on your local computer with Google mlengine commands (for testing):
# then run (after adapting parameters to your specific setup)
#
#   INDIR=<relative path of the location of unpacked train and test data directories>
#   OUTDIR=<relative path of directory where to store model checkpoints and test set predictions>
#
#   gcloud ml-engine local train \
#     --module-name trainer.task \
#     --package-path trainer/ \
#     --job-dir $OUTDIR \
#     -- \
#     --indir $INDIR
#
# for shorter training look at the command line options --train-steps , --train-steps-per-iteration and --eval-steps
# 


#----------------------------------------------------------------------


# Data Loading
import os
import re
from glob import glob

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

#----------
# setup logging for local running
#----------

import logging, sys

# messages written to stdout (print) seem to appear in the log stream
# when running on the ml engine but only at the end ?!
#
# So we use a logger instead of print()
logger = logging.getLogger("task")
logger.setLevel(logging.INFO)

if not os.environ.has_key('CLOUD_ML_JOB_ID'):
    # only set these loggers up when not running on Google ML engine
    # (this is probably not an official way to detect this but
    # should work for the moment)
    #
    # modify the default log format (to include the timestamp also for
    # tensorflow log messages)
    logging.BASIC_FORMAT = "%(levelname)-8s   %(asctime)s   %(name)s   %(message)s"

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(ch)

#----------

# load data from a local file system
def load_data_local(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, wav_data), ...] for train
    [(class_id, user_id, wav_data), ...] for validation

    (compared to the original version of the kernel, this directly returns
    the wav data and thus requires much more memory (for the entire train
    dataset instead of the current train minibatch)

    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            _, wav = wavfile.read(entry)

            sample = (label_id, uid, wav)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    return train, val

#----------

# load data from a Google compute cloud platform storage bucket
def load_data_bucket(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, wav_data), ...] for train
    [(class_id, user_id, wav_data), ...] for validation
    """
    # expects the following files:
    #  <data_dir>/files-noise.csv
    #  <data_dir>/files-train.csv
    #  <data_dir>/wav-noise.npy
    #  <data_dir>/wav-train.npy

    # note that mlengine can't access files with open()
    # we need to use special functions to access these files
    # see e.g. https://stackoverflow.com/a/43242029/288875
    # the code below also works with local files
    from tensorflow.python.lib.io import file_io
    import csv

    possible = set(POSSIBLE_LABELS)
    train, val = [], []

    for wav_file, file_list in (
            ("wav-train.npy", "files-train.csv"),
            ("wav-noise.npy", "files-noise.csv"),
            ):

        wav_file = os.path.join(data_dir, wav_file)
        file_list = os.path.join(data_dir, file_list)

        csv_file_obj = file_io.FileIO(file_list, mode = 'r')
        file_list_reader = csv.DictReader(csv_file_obj)

        wav_file_obj = file_io.FileIO(wav_file, mode = 'r')

        wav_data = np.load(wav_file_obj)

        for row in range(len(wav_data)):

            file_data = next(file_list_reader)

            # this only works for the training set
            label = file_data['label']
            uid   = file_data['speaker']

            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            # get numeric label id
            label_id = name2id[label]

            sample = (label_id, uid, wav_data[row])
            if int(file_data['is_validation']):
                val.append(sample)
            else:
                train.append(sample)

    # end of loop over wav data files / file lists

    return train, val

#----------

def load_data(data_dir):
    if data_dir.startswith("gs://"):
        logger.info("loading data from bucket " + data_dir)
        train, val = load_data_bucket(data_dir)
    else:
        train, val = load_data_local(data_dir)

    logger.info('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

#--------------------

import numpy as np
from scipy.io import wavfile

def data_generator(data, params, mode='train'):

    # data is a list of
    #   (label, user_id, wav_data)

    def generator():
        if mode == 'train':
            # shuffles along the first axis
            np.random.shuffle(data)

        # Feel free to add any augmentation
        for (label_id, uid, wav) in data:
            try:
                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    # note that in contrast to the original kernel,
                    # this only discards samples shorter than one second
                    # when reading from a local file system but not
                    # when reading from the Google cloud platform bucket
                    # (where we have padded shorted samples to one second)
                    continue

                # convert to float32
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20

                for _ in range(samples_per_file):
                    if len(wav) > L:
                        # sample from the longer silence (noise) files
                        # (note that this again is slightly different
                        # from the original kernel since all background
                        # noise wavs are padded to their maximum length)
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0

                    yield dict(
                        target=np.int32(label_id),
                        wav=wav[beg: beg + L],
                    )
            except Exception as err:
                logging.error(err, label_id, uid)

    return generator

#--------------------

import tensorflow as tf
from tensorflow.contrib import layers

def baseline(x, params, is_training):
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(
            x, 16 * (2 ** i), 3, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )
        x = layers.max_pool2d(x, 2, 2)

    # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)
    
    # again conv2d 1x1 instead of dense layer
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])

#----------

from tensorflow.contrib import signal

# features is a dict with keys: tensors from our datagenerator
# labels also were in features, but excluded in generator_input_fn by target_key

def model_handler(features, labels, mode, params, config):
    # Im really like to use make_template instead of variable_scopes and re-usage
    extractor = tf.make_template(
        'extractor', baseline,
        create_scope_now_=True,
    )
    # wav is a waveform signal with shape (16000, )
    wav = features['wav']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        wav,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))
    
    x = tf.stack([amp, phase], axis=3) # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)  # we want to have float32, not float64

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
            'sample': features['sample'], # it's a hack for simplicity
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams,
    )

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--indir',
        help='GCS or local path to data directory containing .wav files',
        nargs=1,
        required=True
    )

    parser.add_argument(
        '--train-steps',
        help='total number of steps to perform during training',
        type = int,
        default = 10000,
    )

    parser.add_argument(
        '--eval-steps',
        help='number of evaluation steps to run (after each iteration ?)',
        type = int,
        default = 200,
    )

    parser.add_argument(
        '--train-steps-per-iteration',
        help='number of training steps per iteration (before evalutaion is run ?)',
        type = int,
        default = 1000,
    )

    parser.add_argument(
        '--allow-memory-growth',
        help='avoid taking all memory on all GPUs',
        default = False,
        action = 'store_true',
    )

    # seems to be required by the gcloud command...
    parser.add_argument(
        '--job-dir',
        help = 'GCS location to write checkpoints and export models',
        required = True,
        nargs = 1,
    )

    args = parser.parse_args()

    #----------
    # hyperparameters
    #----------

    params=dict(
        seed=2018,
        batch_size=64,
        keep_prob=0.5,
        learning_rate=1e-3,
        clip_gradients=15.0,
        use_batch_norm=True,
        num_classes=len(POSSIBLE_LABELS),
    )

    hparams = tf.contrib.training.HParams(**params)

    # load data
    trainset, valset = load_data(args.indir[0])

    # os.makedirs(os.path.join(OUTDIR, 'eval'), exist_ok=True)
    model_dir = args.job_dir[0]

    run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)

    #--------------------

    # it's a magic function :)
    from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

    train_input_fn = generator_input_fn(
        x=data_generator(trainset, hparams, 'train'),
        target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
        batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
        queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
    )

    val_input_fn = generator_input_fn(
        x=data_generator(valset, hparams, 'val'),
        target_key='target',
        batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
        queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
    )


    def _create_my_experiment(run_config, hparams):
        exp = tf.contrib.learn.Experiment(
            estimator                 = create_model(config=run_config, hparams=hparams),
            train_input_fn            = train_input_fn,
            eval_input_fn             = val_input_fn,
            train_steps               = args.train_steps,
            eval_steps                = args.eval_steps,  # read source code for steps-epochs ariphmetics
            train_steps_per_iteration = args.train_steps_per_iteration,
        )
        return exp

    #----------
    # run training
    #----------
    config = tf.ConfigProto()

    if args.allow_memory_growth:
        config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:

        logger.info("starting training")
        tf.contrib.learn.learn_runner.run(
            experiment_fn=_create_my_experiment,
            run_config=run_config,
            schedule="continuous_train_and_eval",
            hparams=hparams)

        #--------------------
        # calculate predictions on test set
        #--------------------

        # now we want to predict!

        if args.indir[0].startswith('gs://'):
            # read test data from a Google cloud platform storage bucket
            # expects the following files:
            #  <data_dir>/files-test.csv
            #  <data_dir>/wav-test.npy

            def test_data_generator(data_dir):

                def generator():
                    from tensorflow.python.lib.io import file_io
                    import csv

                    wav_file = os.path.join(data_dir, "wav-test.npy")
                    file_list = os.path.join(data_dir, "files-test.csv")

                    csv_file_obj = file_io.FileIO(file_list, mode = 'r')
                    file_list_reader = csv.DictReader(csv_file_obj)

                    # open the npy file with the wave data
                    wav_file_obj = file_io.FileIO(wav_file, mode = 'r')
                    wav_data = np.load(wav_file_obj)

                    for row in range(len(wav_data)):

                        fname = next(file_list_reader)['file']

                        yield dict(
                            sample = np.string_(fname),
                            wav = wav_data[row].astype(np.float32) / np.iinfo(np.int16).max,
                            )

                return generator

            tdg = test_data_generator(args.indir[0])

        else:
            # read test data from a local directory
        
            paths = glob(os.path.join(args.indir[0], 'test/audio/*wav'))

            def test_data_generator(data):
                def generator():
                    for path in data:
                        _, wav = wavfile.read(path)

                        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
                        fname = os.path.basename(path)
                        yield dict(
                            sample=np.string_(fname),
                            wav=wav,
                        )

                return generator

            tdg = test_data_generator(paths)

        test_input_fn = generator_input_fn(
            x=tdg,
            batch_size=hparams.batch_size, 
            shuffle=False, 
            num_epochs=1,
            queue_capacity= 10 * hparams.batch_size, 
            num_threads=1,
        )

        model = create_model(config=run_config, hparams=hparams)
        it = model.predict(input_fn=test_input_fn)


        # last batch will contain padding, so remove duplicates
        submission = dict()

        try:
            # Google mlengine does not have tqdm installed
            from tqdm import tqdm
            has_tdqm = True
        except ImportError:
            has_tdqm = False

        logger.info('Starting prediction on test set')

        if has_tdqm:
            for t in tqdm(it, unit = 'files', mininterval = 1.):
                fname, label = t['sample'].decode(), id2name[t['label']]
                submission[fname] = label
        else:
            for t in it:
                fname, label = t['sample'].decode(), id2name[t['label']]
                submission[fname] = label

        from tensorflow.python.lib.io import file_io

        # FileIO works for both local files and Google cloud storage bucket files
        output_fname = os.path.join(model_dir, 'submission.csv')

        logger.info("writing test set predictions to " + output_fname)
        with file_io.FileIO(output_fname, mode = 'w') as fout:
            fout.write('fname,label\n')
            for fname, label in submission.items():
                fout.write('{},{}\n'.format(fname, label))

        #--------------------


