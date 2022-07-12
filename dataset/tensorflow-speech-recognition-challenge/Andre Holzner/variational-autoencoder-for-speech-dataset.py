#!/usr/bin/env python
# 
# Kernel for training a variational autoencoder on Google's ML engine
# for Tensorflow speech recognition challenge
#
# See 
#
#   https://www.kaggle.com/holzner/alex-ozerin-s-estimator-for-google-ml-engine
#
# and
#
#   https://www.kaggle.com/holzner/recipe-for-training-on-google-ml-engine
#
# if you want to run this on Google ML engine

#----------------------------------------------------------------------


# Data Loading
import os
import re
from glob import glob

# the classes we should predict
# in this kernel we restrict ourselves to samples from these classes
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

            if label not in possible:
                # ignore this sample
                continue

            label_id = name2id[label]

            _, wav = wavfile.read(entry)

            sample = (label_id, uid, wav,
                      # for debugging
                      entry
                      )
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

            if label not in possible:
                # ignore files other than the known classes
                continue

            # get numeric label id
            label_id = name2id[label]

            sample = (label_id, uid, wav_data[row], 
                      # for debugging
                      file_data['file']
                      )
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
        for line in data:
            
            if len(line) == 3:
                (label_id, uid, wav) = line
                fname = "???"
            else:
                (label_id, uid, wav, fname) = line
            try:
                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    # pad to one second
                    # (note that in contrast to the original kernel)
                    wav = np.pad(wav, (0, L - len(wav)), 'median')
                elif len(wav) > L:
                    # ignore noise samples
                    continue
                

                # convert to float32
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                yield dict(
                    target=np.int32(label_id),
                    wav=wav,
                    fname = np.string_(fname),
                    )
            except Exception as err:
                logging.error(str(err) + ' ' + str(label_id) + ' ' + str(uid))

    return generator

#--------------------

import tensorflow as tf
from tensorflow.contrib import layers

#----------
# THE MODEL
#
# Autoencoder with convolutional layers at the input/encoding side
# and deconvolutional layers on the output/decoding side
#
# in Alex Ozerin's original kernel there were
#   - a batch normalization at the input
#     (we could still do this but we must make sure we calculate
#      the loss between input and output based on the batch normalized
#      input ?)
#   - convolutional layers
#   - ELU activation functions
#   - batch norm normalizers with the conv2d layers
#   - maxpool 2d layers 

#----------

def baseline(x, params, is_training):
    # x = layers.batch_norm(x, is_training=is_training)

    # instead of using batchnorm, normalize the input spectrograms
    # such that each of them has the smallest element at zero
    # and the largest element at one
    #
    # assume that we never have an all zeros input
    # see https://stackoverflow.com/a/38377600/288875

    # keep track of shapes to have a similar structure
    # on the output (decoder) side
    shapes = [ x.shape.as_list() ]

    # build the encoder (input side) of the autoencoder
    for i in range(4):
        # note that keyword names differ slightly between
        # tensorflow.contrib.layers.conv2d() (used here)
        # and tensorflow.layers.conv2d()

        input_shape = x.shape

        x = layers.conv2d(
            inputs      = x,             
            num_outputs = 16 * (2 ** i),  # number of filters
            kernel_size = 3, 
            stride      = 1, # use stride > 1 instead of maxpool layers
            activation_fn = tf.nn.elu,
            normalizer_fn = layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )

        x = layers.max_pool2d(x, 2, 2)

        shapes.append(x.shape.as_list())

    # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)
    
    # again conv2d 1x1 instead of dense layer
    z_mus                = layers.conv2d(x, params.num_latent_vars, 1, 1, activation_fn = None)
    z_log_sigmas_squared = layers.conv2d(x, params.num_latent_vars, 1, 1, activation_fn = None)

    # z_mus has shape (minibatch_dimension, 1, 1, params.num_latent_vars)

    #----------
    # generate normally distributed random numbers
    #----------
    if is_training:
        eps = tf.random_normal(shape = z_mus.shape,
                               mean = 0,
                               stddev = 1, 
            dtype = tf.float32)
    else:
        # why do we have to do this ???
        the_shape = z_mus.shape.as_list()
        if the_shape[0] is None:
            the_shape[0] = 1

        eps = tf.zeros(shape = the_shape,
                       dtype = tf.float32)

    sigmas = tf.sqrt(tf.exp(z_log_sigmas_squared))
    
    # z = mu + sigma * epsilon
    z = tf.add(z_mus, 
               tf.multiply(sigmas, eps))

    #----------
    # decoder network. We can't make it the exact inverse
    # but it is clearly inspired by the input (encoder) network
    #----------
    x = z

    # start with a dense layer to get the dimensions of the last layer on the input/encoder
    # side

    for shape in shapes:
        if shape[0] is None:
            shape[0] = 1

    target_shape = shapes.pop(-1)

    in_shape = x.shape
    x = tf.layers.dense(
        inputs = x,
        units = np.prod(target_shape[1:]),
        activation = tf.nn.elu,
        )

    x = tf.reshape(x, target_shape)

    # decoding layers:
    #   - could using deconvolutional/transposed convolutional layers
    #   - here using a combination of image upscaling and convolutional
    #     layers like in https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85
    #   
    #
    #
    # on the input (encoder) side the evolutions of the (2D dimensions; # filters) are:
    #   (98,  257;  1) -> (49, 128; 16) -> (24, 64; 32) -> (12, 32;  64) -> (6, 16; 128)
    # 
    # output side models the dimensions in reverse direction

    while shapes:
        shape = shapes.pop(-1)

        x = tf.image.resize_images(images = x, 
                                   size   = shape[1:3],   # output image size
                                   method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        x = layers.conv2d(
            inputs = x, 
            num_outputs = shape[-1], 
            kernel_size = 3, 
            padding = 'SAME', 
            activation_fn = tf.nn.elu,
            normalizer_fn = layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
            )


    # apply an elementwise sigmoid
    # because we normalized the input to be in the range 0..1
    # but don't do this here but do this when calculating
    # the loss instead
    return x, z_mus, z_log_sigmas_squared
        
#----------

from tensorflow.contrib import signal

# features is a dict with keys: tensors from our datagenerator
# labels also were in features, but excluded in generator_input_fn by target_key

def model_handler(features, labels, mode, params, config):

    # use make_template so that variables are the necessary variables are
    # created and used on subsequent calls
    extractor = tf.make_template(
        'extractor',  # name
        baseline,     # the function to wrap
        create_scope_now_ = True, # create now instead of on first call
    )
    
    # wav has shape (minibatch_size, 16000)
    wav = features['wav']

    # we want to compute spectograms by means of short time fourier transform:
    # specgram has shape (minibatch_size, num_time_points, num_frequency_points)
    specgram = signal.stft(
        wav,
        400,  # frame length : 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # frame step:    16000 * 0.010 -- default stride
              # will give 100 transformations
    )

    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))

    # normalize amplitude to range 0..1
    # amp has typically shape (64, 98, 257)

    the_min = tf.expand_dims(tf.expand_dims(tf.reduce_min(amp, axis = [1,2]), axis = 1), axis = 2)
    the_max = tf.expand_dims(tf.expand_dims(tf.reduce_max(amp, axis = [1,2]), axis = 1), axis = 2)

    range = tf.subtract(the_max, the_min)

    amp = tf.div(
        tf.subtract(
            amp, 
            the_min
        ), 
        # protect against the_min == the_max
        tf.maximum(range, 1e-5)
    )

    amp = tf.verify_tensor_all_finite(amp, "normalized spectrum amplitude has some NaNs/Inf")

    # original code included phase
    # and used last ('channel') index to distinguish between amplitude and phase
    # x = tf.stack([amp, phase], axis=3) # shape is [batch_size, time, freq_bins, 2]

    #
    # for the autoencoder we do not use the phase for the moment
    # because it is not straightforward how to define a loss on them

    # add a channel dimension 
    x = tf.expand_dims(amp, axis = 3)
    
    x = tf.to_float(x)  # we want to have float32, not float64

    # pass the input through the encoder and decoder network
    # to get the reconstructed amplitudes
    # (we forget about the phases for the moment)
    reconstructed_amp_logits, z_mus, z_log_sigmas_squared = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    reconstructed_amp = tf.sigmoid(reconstructed_amp_logits)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # in the following, we take the mean over pixels and 

        # loss for VAEs has two components:

        # 1. loss/difference/distance between input and reconstructed output
        reco_loss = tf.reduce_sum(
            # this requires labels to be integers so we can't use it here
            # tf.nn.sparse_softmax_cross_entropy_with_logits(labels = amp, logits = reconstructed_amp_logits)

            # we could drop the sigmoid and use a softmax based on logits but
            # for the moment we implement it by hand so we know what is 'inside'
            - (       x  * tf.log(1e-10 + reconstructed_amp)
               + (1 - x) * tf.log(1e-10 + 1 - reconstructed_amp)),

               axis = [ 1, 2, 3]
            )

        # print which files were involved when a NaN is found 
        reco_loss = tf.cond(
            # condition
            tf.reduce_any(tf.is_nan(reco_loss)),

            # True
            true_fn = lambda: tf.Print(reco_loss, data = [ features['fname'], reco_loss],
                                       message = 'reco_loss has NaN'
                                       ),

            # False
            false_fn = lambda: reco_loss)

        # this will throw an exception if there is a NaN
        reco_loss = tf.verify_tensor_all_finite(reco_loss, "reco_loss has some NaNs/Inf")

        # 2. Kullback-Leibler divergence between values at latent layer
        # and a priori distribution (isotropic Gaussian)
        #
        # from https://jmetzen.github.io/2015-11-27/vae.html
        
        # z_log_sigmas_squared etc. have shape [minibatch_size, 1, 1, num_latent_variables]
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigmas_squared - tf.square(z_mus) - tf.exp(z_log_sigmas_squared), 
                                           axis = [1, 2, 3])

        # this will throw an exception if there is a NaN
        latent_loss = tf.verify_tensor_all_finite(latent_loss, "latent_loss has some NaNs/Inf")

        # take mean over minibatch
        reco_loss = tf.reduce_mean(reco_loss, axis = 0)
        latent_loss = tf.reduce_mean(latent_loss, axis = 0)

        tf.summary.scalar('latent_loss', latent_loss)
        tf.summary.scalar('reco_loss',   reco_loss)

        loss = reco_loss + latent_loss

        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss                   = loss,
            global_step            = tf.train.get_global_step(),
            learning_rate          = params.learning_rate,
            optimizer              = lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn = learning_rate_decay_fn,
            clip_gradients         = params.clip_gradients,
            variables              = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode     = mode,
            loss     = loss,
            train_op = train_op,
        )

    ### if mode == tf.estimator.ModeKeys.EVAL:
    ###     prediction = tf.argmax(logits, axis=-1)
    ###     acc, acc_op = tf.metrics.mean_per_class_accuracy(
    ###         labels, prediction, params.num_classes)
    ###     loss = tf.reduce_mean(
    ###         tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    ###     specs = dict(
    ###         mode=mode,
    ###         loss=loss,
    ###         eval_metric_ops=dict(
    ###             acc=(acc, acc_op),
    ###         )
    ###     )

    if mode == tf.estimator.ModeKeys.PREDICT:

        specs = dict(
            mode = mode,

            # quantities to calculate during the prediction phase
            predictions = {
                "mus"   : z_mus,
                "sigmas": tf.exp(0.5 * z_log_sigmas_squared),
                'sample': features['sample'], 
                }
        )

        
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn = model_handler,
        config   = config,
        params   = hparams,
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
        num_latent_vars=len(POSSIBLE_LABELS),
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
        # calculate mu and sigma vectors on train and test set
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

                    for wav_file, file_list in [
                            ("wav-train.npy", "files-train.csv"),
                            ("wav-test.npy", "files-test.csv")
                            ]:
                        # expand paths
                        wav_file = os.path.join(data_dir, wav_file)
                        file_list = os.path.join(data_dir, file_list)

                        logging.info("opening file list " + file_list)
                        csv_file_obj = file_io.FileIO(file_list, mode = 'r')
                        file_list_reader = csv.DictReader(csv_file_obj)

                        # open the npy file with the wave data
                        logging.info("opening wav file " + wav_file)
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

            paths  = glob(os.path.join(args.indir[0], 'train/audio/*/*wav'))
            paths += glob(os.path.join(args.indir[0], 'test/audio/*wav'))

            def test_data_generator(data):
                def generator():
                    for path in data:

                        _, wav = wavfile.read(path)

                        # exclude background noise, pad shorter samples
                        L = 16000  # be aware, some files are shorter than 1 sec!
                        if len(wav) < L:
                            # pad to one second
                            # (note that in contrast to the original kernel)
                            wav = np.pad(wav, (0, L - len(wav)), 'median')
                        elif len(wav) > L:
                            # noise sample, ignore
                            continue

                        wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                        if 'train' in path:
                            fname = "/".join(path.split('/')[-2:])
                        else:
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
            num_epochs=1, # None would imply infinite loop over the sample
            queue_capacity= 10 * hparams.batch_size, 
            num_threads=1,
        )

        model = create_model(config=run_config, hparams=hparams)
        it = model.predict(input_fn=test_input_fn)


        # last batch will contain padding, so remove duplicates
        submission_mus = dict()
        submission_sigmas = dict()

        try:
            # Google mlengine does not have tqdm installed
            from tqdm import tqdm
            has_tdqm = True
        except ImportError:
            has_tdqm = False

        logger.info('Starting prediction on test set')

        if has_tdqm:
            for t in tqdm(it, unit = 'files', mininterval = 1.):
                fname = t['sample'].decode()
                submission_mus[fname]    = t['mus'][0][0]
                submission_sigmas[fname] = t['sigmas'][0][0]

        else:
            for t in it:
                fname = t['sample'].decode()
                submission_mus[fname]    = t['mus'][0][0]
                submission_sigmas[fname] = t['sigmas'][0][0]

        from tensorflow.python.lib.io import file_io

        # FileIO works for both local files and Google cloud storage bucket files
        output_fname = os.path.join(model_dir, 'submission.csv')

        logger.info("writing results to " + output_fname)
        with file_io.FileIO(output_fname, mode = 'w') as fout:
            header = [ 'fname' ]
            header += [ "mu%02d" % i for i in range(hparams.num_latent_vars) ]
            header += [ "sigma%02d" % i for i in range(hparams.num_latent_vars) ]

            fout.write(','.join(header) + '\n')
            for fname in submission_mus.keys():
                parts = [ fname ]
                parts += submission_mus[fname].tolist()
                parts += submission_sigmas[fname].tolist()
                fout.write(",".join([ str(x) for x in parts]) + "\n")

        #--------------------