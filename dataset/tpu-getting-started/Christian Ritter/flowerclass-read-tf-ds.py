'''module with utility functions'''

import math, re, os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from numpy.random import default_rng


print("Tensorflow version " + tf.__version__)
from functools import partial



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102

AUTO = tf.data.experimental.AUTOTUNE

def decode_image(image_data, IMAGE_SIZE, RESIZE):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    
    if RESIZE:
        target_height, target_width = RESIZE
        image = tf.image.resize_with_pad(
                    image, target_height, target_width #method=ResizeMethod.BILINEAR,
                    #antialias=False
                )
        
    return image

def read_labeled_tfrecord(example, IMAGE_SIZE, RESIZE, with_id=False):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'], IMAGE_SIZE, RESIZE)
    label = tf.cast(example['class'], tf.int32)
    depth = tf.constant(104)

    one_hot_encoded = tf.one_hot(indices=label, depth=depth)
    
    if not with_id:
        return image, one_hot_encoded # returns a dataset of (image, label) pairs
    else:
        image_id = tf.cast(example['id'], tf.string)
        return image, one_hot_encoded, image_id
    
    
def read_unlabeled_tfrecord(example, IMAGE_SIZE, RESIZE):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'], IMAGE_SIZE, RESIZE)
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, IMAGE_SIZE, RESIZE, labeled=True, ordered=False, with_id=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    read_labeled_tfrecord2 = partial(read_labeled_tfrecord, IMAGE_SIZE=IMAGE_SIZE, RESIZE=RESIZE, with_id=with_id)
    read_unlabeled_tfrecord2 = partial(read_unlabeled_tfrecord, IMAGE_SIZE=IMAGE_SIZE, RESIZE=RESIZE)
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)# AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord2 if labeled else read_unlabeled_tfrecord2, num_parallel_calls=1)#AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO)
    # statement in the next function (below), this happens essentially
    # for free on TPU. Data pipeline code is executed on the "CPU"
    # part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

def get_training_dataset(TRAINING_FILENAMES, BATCH_SIZE, IMAGE_SIZE, RESIZE, with_id):
    dataset = load_dataset(TRAINING_FILENAMES, IMAGE_SIZE, RESIZE, labeled=True, with_id=with_id)
    #dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(VALIDATION_FILENAMES, BATCH_SIZE, IMAGE_SIZE, RESIZE, with_id, ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, IMAGE_SIZE, RESIZE, labeled=True, ordered=ordered, with_id=with_id)
    dataset = dataset.batch(BATCH_SIZE)
    #dataset = dataset.cache()
    #dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(TEST_FILENAMES, BATCH_SIZE, IMAGE_SIZE, RESIZE, ordered=False):
    dataset = load_dataset(TEST_FILENAMES, IMAGE_SIZE, RESIZE, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def get_datasets(BATCH_SIZE, IMAGE_SIZE, RESIZE, tpu=False, with_id=False):

    if not tpu:
        data_root = "../input/tpu-getting-started"
    else:
        #data_root = GCS_DS_PATH
        print('tpu not implemented yet')
        return 

    if 512 in IMAGE_SIZE:

        data_path = data_root + '/tfrecords-jpeg-512x512'
        #IMAGE_SIZE = [512, 512]
    elif 224 in IMAGE_SIZE:
        data_path = data_root + "/tfrecords-jpeg-224x224"
        #IMAGE_SIZE = [224, 224]
    elif 331 in IMAGE_SIZE:
        data_path = data_root + "/tfrecords-jpeg-331x331"
        #IMAGE_SIZE = [331, 331]
    elif 192 in IMAGE_SIZE:
        data_path = data_root + "/tfrecords-jpeg-192x192"
        #IMAGE_SIZE = [192, 192]      
    else:
        print("wrong image size")
        return

    TRAINING_FILENAMES = tf.io.gfile.glob(data_path + '/train/*.tfrec')
    VALIDATION_FILENAMES = tf.io.gfile.glob(data_path + '/val/*.tfrec')
    TEST_FILENAMES = tf.io.gfile.glob(data_path + '/test/*.tfrec') 


    ds_train = get_training_dataset(TRAINING_FILENAMES, BATCH_SIZE, IMAGE_SIZE, RESIZE, with_id=with_id)
    ds_valid = get_validation_dataset(VALIDATION_FILENAMES, BATCH_SIZE, IMAGE_SIZE, RESIZE, with_id=with_id)
    ds_test = get_test_dataset(TEST_FILENAMES, BATCH_SIZE, IMAGE_SIZE, RESIZE)

    print("Training:", ds_train)
    print ("Validation:", ds_valid)
    print("Test:", ds_test)
    
    return ds_train, ds_valid, ds_test


def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case,
                                     # these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is
    # the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    


def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


#### image analysis

def display_batch_of_images(databatch, predictions=None, FIGSIZE=13, image_ids=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square
    # or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows + 1
        
    # size and spacing
    #FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        if image_ids is not None: title = title + "\n" + image_ids[i]
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_batch_by_class(files, name = 'iris', top_n= 10, FIGSIZE=13):
    
    
    class_name_mapping = {i:name for i, name in enumerate(CLASSES)}
    inverse_class_name_mapping = {class_name_mapping[i]: i for i in class_name_mapping}    
    class_idx = inverse_class_name_mapping[name]
    print(class_idx)
    
    # get position of class images in dataset
    sample_idx = []
    
    ds = load_dataset(files, labeled=True, IMAGE_SIZE=(224, 224), RESIZE=None)
    ds = ds.batch(1)
    for i, (img, label) in tqdm(enumerate(ds)):
        #print(label)
        if label.numpy()[0].argmax() == class_idx:
            sample_idx.append(i)
            
    print(f"found {len(sample_idx)} images and take sample of {top_n} images")
    # choose randomly top_n images
    rng = default_rng(42)
    sample_idx_shuffled = sample_idx.copy()
    rng.shuffle(sample_idx_shuffled)
    top_n_sample = sample_idx_shuffled[:top_n]

    ds = load_dataset(files, labeled=True, IMAGE_SIZE=(224, 224), RESIZE=None)
    ds = ds.batch(1)
    # get thte images for each data point
    images_class = []
    tmp = []
    for i, (img, label) in tqdm(enumerate(ds)):
        if i in top_n_sample:
            images_class.append(img)
            tmp.append(label)

    batch = tf.stack([tf.squeeze(img) for img in images_class]), tf.stack([class_idx for i in range(len(images_class))])
    
    display_batch_of_images(batch, FIGSIZE=FIGSIZE)

