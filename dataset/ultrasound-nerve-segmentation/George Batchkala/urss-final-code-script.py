# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# check_pars
# ======================================================================================================================
########################################################################################################################
# read-only file!!!

# standard-module imports
from keras.optimizers import Adam


def check_dict_subset(subset, superset):
    """Checks if one nested dictionary is a subset of another

    :param subset: subset dictionary
    :param superset: superset dictionary
    :return: if failed: gives helpful print statements and assertion error
             if successful, prints 'Your parameter choice is valid'
    """
    print("superset keys:", superset.keys())
    print("subset keys:", subset.keys())
    assert all(item in superset.keys() for item in subset.keys())
    print("Subset keys is a subset of superset keys", all(item in superset.keys() for item in subset.keys()))
    for key in subset.keys():
        print("superset key items:", superset[key])
        print("subset key items:", subset[key])
        if type(superset[key]) == dict:
            assert type(subset[key]) == type(superset[key])
            check_dict_subset(subset[key], superset[key])
        elif type(superset[key]) == list:
            assert subset[key] in superset[key]
            print("subset[key] item:", subset[key], " is in superset[key] items:", superset[key])
        else:
            print("Something went wrong. Uncomment the print statements in check_dict_subset() for easier debugging.")
            return type(superset[key]), superset[key]

    return 'Your parameter choice is valid'


# Only change ALLOWED_PARS if adding new functionality
ALLOWED_PARS = {
    'outputs': [1, 2],
    'activation': ['elu', 'relu'],
    'pooling_block': {
        'trainable': [True, False]},
    'information_block': {
        'inception': {
            'v1': ['a', 'b'],
            'v2': ['a', 'b', 'c'],
            'et': ['a', 'b']},
        'convolution': {
            'simple': ['not_normalized', 'normalized'],
            'dilated': ['not_normalized', 'normalized']}},
    'connection_block': ['not_residual', 'residual']
}

# for reference: in combination, these parameter choice showed the best performance
BEST_OPTIMIZER = Adam(lr=0.0045)
BEST_PARS = {
    'outputs': 2,
    'activation': 'elu',
    'pooling_block': {'trainable': True},
    'information_block': {'inception': {'v2': 'b'}},
    'connection_block': 'residual'
}

print(check_dict_subset(BEST_PARS, ALLOWED_PARS))


########################################################################################################################
# ======================================================================================================================
# configuration
# ======================================================================================================================
########################################################################################################################

# standard-module imports
from keras.optimizers import Adam

# # separate-module imports
# from check_pars import ALLOWED_PARS, check_dict_subset

# look up the format and the available parameters
print(ALLOWED_PARS)

# The result is very sensitive to the choice of the Learning Rate parameter  of the optimizer
# DO NOT CHANGE THE NAME, you can change the parameters
OPTIMIZER = Adam(lr=0.0045)

# DO NOT CHANGE THE NAME, you can change the parameters
PARS = {
    'outputs': 1,
    'activation': 'relu',
    'pooling_block': {'trainable': False},
    'information_block': {'convolution': {'simple': 'normalized'}},
    'connection_block': 'not_residual'
}

# DO NOT REMOVE THESE LINES, they checks if your parameter choice is valid
assert PARS.keys() == ALLOWED_PARS.keys()
print(check_dict_subset(PARS, ALLOWED_PARS))


########################################################################################################################
# ======================================================================================================================
# data
# ======================================================================================================================
########################################################################################################################

# ======================================================================================================================
# Set-up
# ======================================================================================================================

# standard-module imports
import os
import numpy as np
from skimage.io import imread

# Input data files are available in the "../input/" directory.

_dir = os.path.abspath('')
os.chdir(_dir)
print(_dir)

print(os.listdir(_dir))
print(os.listdir("../input"))
print(os.listdir("../input/ultrasound-nerve-segmentation"))

# data
data_path = os.path.join('../input/ultrasound-nerve-segmentation', '')
preprocess_path = os.path.join(_dir, 'np_data')

if not os.path.exists(preprocess_path):
    os.mkdir(preprocess_path)
print(os.listdir(_dir))

# train data
img_train_path = os.path.join(preprocess_path, 'imgs_train.npy')
img_train_mask_path = os.path.join(preprocess_path, 'imgs_mask_train.npy')
img_train_patients = os.path.join(preprocess_path, 'imgs_patient.npy')
img_nerve_presence = os.path.join(preprocess_path, 'nerve_presence.npy')

# test data
img_test_path = os.path.join(preprocess_path, 'imgs_test.npy')
img_test_id_path = os.path.join(preprocess_path, 'imgs_id_test.npy')

# image dimensions
image_rows = 420
image_cols = 580


# ======================================================================================================================
# Functions for test and train data creation, storage and access
# ======================================================================================================================

def load_test_data():
    """Load test data from a .npy file.

    :return: np.array with test data.
    """
    print('Loading test data from %s' % img_test_path)
    imgs_test = np.load(img_test_path)
    return imgs_test


def load_test_ids():
    """Load test ids from a .npy file.

    :return: np.array with test ids. Shape (samples, ).
    """
    print('Loading test ids from %s' % img_test_id_path)
    imgs_id = np.load(img_test_id_path)
    return imgs_id


def load_train_data():
    """Load train data from a .npy file.

    :return: np.array with train data.
    """
    print('Loading train data from %s and %s' % (img_train_path, img_train_mask_path))
    imgs_train = np.load(img_train_path)
    imgs_mask_train = np.load(img_train_mask_path)
    return imgs_train, imgs_mask_train


def load_patient_num():
    """Load the array with patient numbers from a .npy file

    :return: np.array with patient numbers
    """
    print('Loading patient numbers from %s' % img_train_patients)
    return np.load(img_train_patients)


def load_nerve_presence():
    """Load the array with binary nerve presence from a .npy file

    :return: np.array with patient numbers
    """
    print('Loading nerve presence array from %s' % img_nerve_presence)
    return np.load(img_nerve_presence)


def get_patient_nums(string):
    """Create a tuple (patient, photo) from image-file name patient_photo.tif

    :param string: image-file name in string format: patient_photo.tif
    :return: a tuple (patient, photo)

    >>> get_patient_nums('32_50.tif')
    (32, 50)
    """
    patient, photo = string.split('_')
    photo = photo.split('.')[0]
    return int(patient), int(photo)


def get_nerve_presence(mask_array):
    """Create an array specifying nerve presence on each of the masks in the mask_array

    :param mask_array: 4D tensor of a shape (samples, rows, cols, channels=1) with masks
    :return:
    """
    print("type(mask_array):", type(mask_array))
    print("mask_array.shape:", mask_array.shape)
    return np.array([int(np.sum(mask_array[i, :, :, 0]) > 0) for i in range(mask_array.shape[0])])


def create_train_data():
    """
    Create an np.array with patient numbers and save it into a .npy file.
    Create an np.array with train images and save it into a .npy file.
    Create an np.array with train masks and save it into a .npy file.

    The np.array with patient numbers will have shape (samples, ).
        So for each train image saved, the patient number will be recorded exactly in the same order the
        images were saved.
    The np.array with train images will have shape (samples, rows, cols, channels).
    The np.array with train masks will have shape (samples, rows, cols, channels).
        The masks are saved in the same order as the images.
    """
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    i = 0
    print('Creating training images...')
    img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:

        # With "continue" skip the mask image in the iteration because the mask will be saved together with
        # the image, when we get the image in one of the next iterations. This guarantees that the images,
        # masks and corresponding patient numbers are all saved in the correct order.
        if 'mask' in image_name:
            continue

        # we got to this point, meaning that image_name is a name of a training image and not a mask.

        # recreate the mask's name fot this image
        # noinspection PyTypeChecker
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # get the patient number of the image
        patient_num = image_name.split('_')[0]
        # read the image itself to an np.array
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        # read the corresponding mask to an np.array
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        imgs[i, :, :, 0] = img
        imgs_mask[i, :, :, 0] = img_mask
        img_patients[i] = patient_num
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    # saving patient numbers, train images, train masks, nerve presence
    np.save(img_train_patients, img_patients)
    np.save(img_train_path, imgs)
    np.save(img_train_mask_path, imgs_mask)
    np.save(img_nerve_presence, get_nerve_presence(imgs_mask))

    print('Saving to .npy files done.')


def create_test_data():
    """
    Create an np.array with test data and save it into a .npy file.
    Create an np.array with ids for all images and save it into a .npy file.

    The np.array with test data will have shape (samples, rows, cols, channels).
    The np.array with test data ids will have shape (samples,). Each image id will be a number
    corresponding to the number in a test image name. For example image '8.tif' will have 8 as its image id.
    """
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('Creating test images...')
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)

        imgs[i, :, :, 0] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(img_test_path, imgs)
    np.save(img_test_id_path, imgs_id)
    print('Saving to .npy files done.')


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    create_train_data()
    create_test_data()

# checking what is in the directory
print(os.listdir(preprocess_path))


########################################################################################################################
# ======================================================================================================================
# metric
# ======================================================================================================================
########################################################################################################################
# needed for u_model

# standard-module imports
import numpy as np
from keras import backend as K  # tensorflow backend


def dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    We need to add smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of
    giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    mask_1_flat = K.flatten(mask_1)
    mask_2_flat = K.flatten(mask_2)

    # for pixel values in {0, 1} multiplication is the intersection of masks
    intersection = K.sum(mask_1_flat * mask_2_flat)
    return (2. * intersection + smooth) / (K.sum(mask_1_flat) + K.sum(mask_2_flat) + smooth)


def dice_coef_loss(mask_pred, mask_true):
    """Calculate dice coefficient loss, when comparing predicted mask for an image with the true mask

    :param mask_pred: predicted mask
    :param mask_true: true mask
    :return: dice coefficient loss
    """
    return -dice_coef(mask_pred, mask_true)


def np_dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Used for testing on artificially generated np.arrays

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Need smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    tr = mask_1.flatten()
    pr = mask_2.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    a = np.random.random((420, 100))
    b = np.random.random((420, 100))
    res = np_dice_coef(a, b)
    print(res)


########################################################################################################################
# ======================================================================================================================
# u_model_blocks
# ======================================================================================================================
########################################################################################################################
# needed for u_model

# standard-module imports
from keras.layers import add, concatenate, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU


# ======================================================================================================================
# utility blocks needed for internal performance
# ======================================================================================================================

def NConv2D(filters, kernel_size, strides=(1, 1), padding='valid', dilation_rate=1,
            activation=None, kernel_initializer='glorot_uniform'):
    """Create a (Normalized Conv2D followed by a chosen activation) function
    Conv2D -> BatchNormalization -> activation()

    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
    convolution)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
                    and width. Can be a single integer to specify the same value for all spatial dimensions.
                    Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param dilation_rate: an integer or tuple/list of a single integer, specifying the dilation rate
                    to use for dilated convolution. Currently, specifying any dilation_rate value != 1
                    is incompatible with specifying any strides value != 1
    :param activation:  string, one of 'elu' or 'relu' or None (case-sensitive),
                        specifies activation function to be performed after BatchNormalization
    :param kernel_initializer: Initializer for the kernel weights matrix (see initializers in keras documentation)
    :return: a function, combined of 2D Convolution, followed by BatchNormalization across filters,
             and specified activation in that order
    """
    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    def f(_input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      dilation_rate=dilation_rate, kernel_initializer=kernel_initializer)(_input)
        norm = BatchNormalization(axis=3)(conv)
        return actv()(norm)

    return f


# needed for rblock (residual block)
def _shortcut(_input, residual):
    stride_width = _input._keras_shape[1] / residual._keras_shape[1]
    stride_height = _input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == _input._keras_shape[3]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual._keras_shape[3], kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          kernel_initializer="he_normal", padding="valid")(_input)

    return add([shortcut, residual])


def rblock(inputs, filters, kernel_size, padding='valid', activation=None, scale=0.1):
    """Create a scaled Residual block connecting the down-path and the up-path of the u-net architecture

    Activations are scaled by a constant to prevent the network from dying. Usually is set between 0.1 and 0.3. See:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output convolution filters)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation:  string, one of 'elu' or 'relu' or None (case-sensitive),
                        specifies activation function to use everywhere in the block
    :param scale: scaling factor preventing the network from dying out
    :return: 4D tensor (samples, rows, cols, channels) output of a residual block, given inputs
    """
    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    residual = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    residual = BatchNormalization(axis=3)(residual)
    residual = Lambda(lambda x: x * scale)(residual)
    res = _shortcut(inputs, residual)
    return actv()(res)


# ======================================================================================================================
# information blocks
# ======================================================================================================================

def convolution_block(inputs, filters, kernel_size=(3, 3), padding='valid', activation=None,
                      version='normalized', pars={}, allowed_pars={}):
    """Create a version of a convolution block.

    Versions: with and without batch-normalization after convolutions.

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of the convolution block, one of 'not_normalized', 'normalized' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of a convolution block, given inputs
    """
    assert activation in ['relu', 'elu', None]

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('convolution').get('simple') == ['not_normalized',
                                                                                          'normalized']
    # keep version argument if need to use without PARS
    assert version in ['not_normalized', 'normalized']
    # setting the version from pars
    if pars.get('information_block').get('convolution').get('simple') is not None:
        version = pars.get('information_block').get('convolution').get('simple')

    if version == 'normalized':
        conv1 = NConv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
        return NConv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv1)
    else:
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
        return Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv1)


def dilated_convolution_block(inputs, filters, kernel_size=(3, 3), padding='valid', activation=None,
                              version='normalized', pars={}, allowed_pars={}):
    """Create a version of a dilated-convolution block.

    Versions: with and without batch-normalization after dilated convolutions.

    See more about dilated convolutions:
    https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of the dilated-convolution block, one of 'not_normalized', 'normalized' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of a dilated-convolution block, given inputs
    """
    assert activation in ['relu', 'elu', None]

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('convolution').get('dilated') == ['not_normalized',
                                                                                           'normalized']
    # keep version argument if need to use without PARS
    assert version in ['not_normalized', 'normalized']
    # setting the version from pars
    if pars.get('information_block').get('convolution') is not None:
        version = pars.get('information_block').get('convolution')

    if version == 'normalized':
        conv1 = NConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                        dilation_rate=2, activation=activation)(inputs)
        return NConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                       dilation_rate=1, activation=activation)(conv1)
    else:
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                       dilation_rate=2, activation=activation)(inputs)
        return Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                      dilation_rate=1, activation=activation)(conv1)


def inception_block_v1(inputs, filters, activation=None, version='b', pars={}, allowed_pars={}):
    """Create a version of v1 inception block described in:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Create an inception block described in v1, sections 'a' (for naive version), or 'b' (with dimension reduction)
    Each version has 4 verticals in their structure. See the link above.

    For all versions, verticals 1 and 2 of the block start with 2D convolution, which:
        reduces the number of input filters to next convolutions (to make computation cheaper)
        uses (1, 1) kernels, no Normalization
        is NOT normalized
        is followed by specified activation
    For all versions, verticals 1, 2, 3:
        the final convolution layer is not normalised and not activated since it will be dene after concatenation
    Vertical 4 is just a Conv2D. Its gets normalized and activated after being concatenated with
        outputs of other verticals.
    The concatenated output of the verticals is normalised and then activated with a given activation

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
    convolution).
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of inception block, one of 'a', 'b' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of an inception block, given inputs
    """

    assert filters % 16 == 0

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('inception').get('v1') == ['a', 'b']
    # keep version argument if need to use without PARS
    assert version in ['a', 'b']
    # setting the version from pars
    if pars.get('information_block').get('inception').get('v1') is not None:
        version = pars.get('information_block').get('inception').get('v1')

    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    # vertical 1
    if version == 'a':
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    else:
        c1_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), padding='same',
                      activation=activation, kernel_initializer='he_normal')(inputs)
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal')(c1_1)

    # vertical 2
    if version == 'a':
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    else:
        c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), padding='same',
                      activation=activation, kernel_initializer='he_normal')(inputs)
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(c2_1)

    # vertical 3
    p3_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    if version == 'b':
        c3 = Conv2D(filters=filters // 8, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(p3_1)
    else:
        c3 = p3_1

    # vertical 4
    c4_1 = Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    c4 = c4_1

    # concatenating verticals together, normalizing and applying activation
    result = concatenate([c1, c2, c3, c4], axis=3)
    result = BatchNormalization(axis=3)(result)
    result = actv()(result)
    return result


def inception_block_v2(inputs, filters, activation=None, version='b', pars={}, allowed_pars={}):
    """Create a version of v1 inception block described in:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Create an inception block described in v2, sections 'a', 'b', or 'c'
    Each version has 4 verticals in their structure. See the link above.

    For all versions, verticals 1 and 2 of the block start with 2D convolution, which:
        reduces the number of input filters to next convolutions (to make computation cheaper)
        uses (1, 1) kernels, no Normalization
        is NOT normalized
        is followed by specified activation
    For all versions, verticals 1, 2, 3:
        the middle convolutions use NConv2D with given activation, see its docstring
        the final convolution layer is not normalised and not activated since it will be dene after concatenation
    Vertical 4 is just a Conv2D. Its gets normalized and activated after being concatenated with
        outputs of other verticals.
    The concatenated output of the verticals is normalised and then activated with a given activation

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of inception block, one of 'a', 'b', 'c' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of an inception block, given inputs
    """
    assert filters % 16 == 0

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('inception').get('v2') == ['a', 'b', 'c']
    # keep version argument if need to use without PARS
    assert version in ['a', 'b', 'c']
    # setting the version from pars
    if pars.get('information_block').get('inception').get('v2') is not None:
        version = pars.get('information_block').get('inception').get('v2')

    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    # vertical 1
    c1_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'a':
        c1_2 = NConv2D(filters=filters // 8, kernel_size=3, padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1 = Conv2D(filters=filters // 8, kernel_size=3, padding='same', kernel_initializer='he_normal')(c1_2)
    elif version == 'b':
        c1_2 = NConv2D(filters=filters // 8, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1_3 = NConv2D(filters=filters // 8, kernel_size=(3, 1), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_2)
        c1_4 = NConv2D(filters=filters // 8, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_3)
        c1 = Conv2D(filters=filters // 8, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c1_4)
    else:
        c1_2 = NConv2D(filters=filters // 8, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1_3 = NConv2D(filters=filters // 8, kernel_size=3, padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_2)
        c1_41 = Conv2D(filters=filters // 8, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal')(c1_3)
        c1_42 = Conv2D(filters=filters // 8, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c1_3)
        c1 = concatenate([c1_41, c1_42], axis=3)

    # vertical 2
    c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'a':
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(c2_1)
    elif version == 'b':
        c2_2 = NConv2D(filters=filters // 2, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c2_1)
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c2_2)
    else:
        c2_21 = Conv2D(filters=filters // 2, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal')(c2_1)
        c2_22 = Conv2D(filters=filters // 2, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c2_1)
        c2 = concatenate([c2_21, c2_22], axis=3)

    # vertical 3
    p3_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    c3 = Conv2D(filters=filters // 8, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(p3_1)

    # vertical 4
    c4 = Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # concatenating verticals together, normalizing and applying activation
    result = concatenate([c1, c2, c3, c4], axis=3)
    result = BatchNormalization(axis=3)(result)
    result = actv()(result)
    return result


def inception_block_et(inputs, filters, activation='relu', version='b', pars={}, allowed_pars={}):
    """Create an inception block with 2 options.
    For intuition read, parts v1 and v2:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Each version/option has 4 verticals in their structure. See the link above.
    Default option: version='b'
        Create an inception block close to one described in v2, but keeps 5 as a factor for some convolutions
    Alternative option: version='a'
        Create an inception block described in v1, section


    Function author Edward Tyantov. That's why the name: inception_block_et.
    My modifications

        use version='a' instead of split=False
        use version='b' instead of split=True

        change default to version='b', aka split=True

        swap: Conv2D -> BatchNormalization -> activation
        to:   NConv2D blocks. See NConv2D documentation for them.

        swap: Conv2D -> activation
        to:   Conv2D -> Conv2D(activation=activation)

        change the order of the verticals to coincide with v2_paper notation

        change names of the outputs of the block verticals to c1, c2, c3, c4

        use 'result' instead of 'res' to avoid confusion with residuals

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param activation: activation function to use everywhere in the block
    :param version: version of inception block
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of an inception block, given inputs
    """
    assert filters % 16 == 0

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('inception').get('et') == ['a', 'b']
    # keep version argument if need to use without PARS
    assert version in ['a', 'b']
    # setting the version from pars
    if pars.get('information_block').get('inception').get('et') is not None:
        version = pars.get('information_block').get('inception').get('et')

    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    # vertical 1
    c1_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'b':
        c1_2 = NConv2D(filters=filters // 8, kernel_size=(1, 5), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same')(c1_2)
    else:
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(c1_1)

    # vertical 2
    c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'b':
        c2_2 = NConv2D(filters=filters // 2, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c2_1)
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 1), kernel_initializer='he_normal', padding='same')(c2_2)
    else:
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c2_1)

    # vertical 3
    p3_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    c3 = Conv2D(filters=filters // 8, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(p3_1)

    # vertical 4
    c4 = Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # concatenating verticals together, normalizing and applying activation
    result = concatenate([c1, c2, c3, c4], axis=3)
    result = BatchNormalization(axis=3)(result)
    result = actv()(result)
    return result


# ======================================================================================================================
# Combining blocks, allowing to use different blocks from before
# ======================================================================================================================

def pooling_block(inputs, filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None,
                  pool_size=(2, 2), trainable=True, pars={}, allowed_pars={}):
    """Function returning the output of one of the pooling blocks.

    Allows not to make different versions of the u-net in terms of how pooling operation is performed:
        1) trainable (default): through NConv2D custom function, see its documentation
        2) non-trainable (alternative): through MaxPooling operation

    To get the expected behaviour when changing 'trainable' assert strides == pool_size

    Parameters starting with p_ are only to be used for (trainable=False) MaxPooling2D
    Parameters starting with c_ are only to be used for (trainable=True) MaxPooling2D

    :param inputs: 4D tensor (samples, rows, cols, channels)
    :param filters:     NConv2D argument, filters
    :param kernel_size: NConv2D argument, kernel_size
    :param strides:     NConv2D argument, strides
    :param padding:     NConv2D/MaxPooling2D argument, padding
    :param activation:  NConv2D argument, activation
    :param pool_size:   MaxPooling2D argument, pool_size

    :param trainable: boolean specifying the version of a pooling block with default behaviour
        trainable=True: NConv2D(inputs._keras_shape[3], kernel_size=kernel_size, strides=strides, padding=padding)(
        inputs)
        trainable=False: MaxPooling2D(pool_size=pool_size)(inputs)
    :param pars: dictionary of parameters passed to u-net, determines the version of the block
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters

    :return: 4D tensor (samples, rows, cols, channels) output of a pooling block
    """
    # checking that the allowed trainable parameters did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('pooling_block').get('trainable') == [True, False]
    # keep trainable argument if need to use without PARS
    assert trainable in [True, False]

    # setting the version from pars
    if pars.get('pooling_block').get('trainable') is not None:
        trainable = pars.get('pooling_block').get('trainable')

    # returning block's output
    if trainable:
        return NConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       padding=padding, activation=activation)(inputs)
    else:
        return MaxPooling2D(pool_size=pool_size, padding=padding)(inputs)


def information_block(inputs, filters, kernel_size=(3, 3), padding='valid', activation=None,
                      block='inception', block_type='v2', version='b', pars={}, allowed_pars={}):
    """Function returning the output of one of the information blocks.

    :param inputs: 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation: string, specifies activation function to use everywhere in the block

    Next 3 parameters are there to be able to leave 'pars' and 'allowed_pars' empty
    :param block:       one of 'inception' or 'convolution' (case-sensitive)
    :param block_type:  if block == 'inception', one of 'v1', 'v2', 'et' (case-sensitive)
                        if block == 'convolution': one of 'simple', 'dilated' (case-sensitive)
    :param version:     version of a block to use

    :param pars: dictionary of parameters passed to u-net, determines the version of the block
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters

    :return: 4D tensor (samples, rows, cols, channels) output of a information block
    """
    # getting which block, block_type, version to use as the information block
    if pars.get('information_block') is not None:
        block = list(pars.get('information_block').keys())[0]
        block_type = list(pars.get('information_block').get(block).keys())[0]
        version = pars.get('information_block').get(block).get(block_type)

    # inception block
    if block == 'inception':
        if block_type == 'v1':
            return inception_block_v1(inputs=inputs, filters=filters, activation=activation,
                                      version=version, pars=pars, allowed_pars=allowed_pars)
        elif block_type == 'v2':
            return inception_block_v2(inputs=inputs, filters=filters, activation=activation,
                                      version=version, pars=pars, allowed_pars=allowed_pars)
        else:
            return inception_block_et(inputs=inputs, filters=filters, activation=activation,
                                      version=version, pars=pars, allowed_pars=allowed_pars)
    # convolution block
    else:
        if block_type == 'simple':
            return convolution_block(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                     padding=padding, activation=activation,
                                     version=version, pars=pars, allowed_pars=allowed_pars)
        else:
            return dilated_convolution_block(inputs=inputs, filters=filters,
                                             kernel_size=kernel_size, padding=padding,
                                             activation=activation, version=version,
                                             pars=pars, allowed_pars=allowed_pars)


def connection_block(inputs, filters, padding='valid', activation=None,
                     version='residual', pars={}, allowed_pars={}):
    """Function returning the output of one of the connection block.

    :param inputs: 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation:  string, one of 'elu' or 'relu' or None (case-sensitive),
                        specifies activation function to use everywhere in the block

    Version parameter is there to be able to leave 'pars' and 'allowed_pars' empty
    :param version: one of 'not_residual' or 'residual', version of a block to use

    :param pars: dictionary of parameters passed to u-net, determines the version of the block
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters

    :return: 4D tensor (samples, rows, cols, channels) output of a connection block
    """
    # checking that the allowed trainable parameters did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('connection_block') == ['not_residual', 'residual']
    # keep trainable argument if need to use without PARS
    assert version in ['not_residual', 'residual']
    # setting the version from pars
    if pars.get('connection_block') is not None:
        version = pars.get('connection_block')

    if version == 'residual':
        return rblock(inputs=inputs, filters=32, kernel_size=(1, 1), padding='same', activation=activation)
    else:
        return Conv2D(filters=filters, kernel_size=(2, 2), padding=padding, kernel_initializer='he_normal')(inputs)


########################################################################################################################
# ======================================================================================================================
# u_model
# ======================================================================================================================
########################################################################################################################
# needed for train

# standard-module imports
import numpy as np
from keras.layers import Input, concatenate, Conv2D, UpSampling2D, Dense
from keras.layers import Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

# # separate-module imports
# from metric import dice_coef, dice_coef_loss
# from u_model_blocks import pooling_block, connection_block, information_block
# from configuration import ALLOWED_PARS, PARS


IMG_ROWS, IMG_COLS = 80, 112
K.set_image_data_format('channels_last')  # (number of images, rows per image, cols per image, channels)


# ======================================================================================================================
# U-net with Inception blocks, Normalised 2D Convolutions instead of Maxpooling
# ======================================================================================================================

def get_unet_customised(optimizer, pars=PARS, allowed_pars=ALLOWED_PARS):
    """
    Creating and compiling the U-net

    This version is fully customisable by choosing pars argument

    :param optimizer: specifies the optimiser for the u-net, e.g. Adam, RMSProp, etc.
    :param pars: optional, dictionary of parameters passed to customise the U-net
    :param allowed_pars: optional, dictionary of parameters allowed to be passed to customise the U-net
    :return: compiled u-net, Keras.Model object
    """

    # string, activation function
    activation = pars.get('activation')

    # input
    inputs = Input((IMG_ROWS, IMG_COLS, 1), name='main_input')
    print('inputs:', inputs._keras_shape)

    #
    # down the U-net
    #

    conv1 = information_block(inputs, 32, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv1', conv1._keras_shape)
    pool1 = pooling_block(inputs=conv1, filters=32, activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('pool1', pool1._keras_shape)
    pool1 = Dropout(0.5)(pool1)
    print('pool1', pool1._keras_shape)

    conv2 = information_block(pool1, 64, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv2', conv2._keras_shape)
    pool2 = pooling_block(inputs=conv2, filters=64, activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('pool2', pool2._keras_shape)
    pool2 = Dropout(0.5)(pool2)
    print('pool2', pool2._keras_shape)

    conv3 = information_block(pool2, 128, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv3', conv3._keras_shape)
    pool3 = pooling_block(inputs=conv3, filters=128, activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('pool3', pool3._keras_shape)
    pool3 = Dropout(0.5)(pool3)
    print('pool3', pool3._keras_shape)

    conv4 = information_block(pool3, 256, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv4', conv4._keras_shape)
    pool4 = pooling_block(inputs=conv4, filters=256, activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('pool4', pool4._keras_shape)
    pool4 = Dropout(0.5)(pool4)
    print('pool4', pool4._keras_shape)

    #
    # bottom level of the U-net
    #
    conv5 = information_block(pool4, 512, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv5', conv5._keras_shape)
    conv5 = Dropout(0.5)(conv5)
    print('conv5', conv5._keras_shape)

    #
    # auxiliary output for predicting probability of nerve presence
    #
    if pars['outputs'] == 2:
        pre = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid')(conv5)
        pre = Flatten()(pre)
        aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)

    #
    # up the U-net
    #

    after_conv4 = connection_block(conv4, 256, padding='same', activation=activation,
                                   pars=pars, allowed_pars=allowed_pars)
    print('after_conv4', after_conv4._keras_shape)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=3)
    conv6 = information_block(up6, 256, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv6', conv6._keras_shape)
    conv6 = Dropout(0.5)(conv6)
    print('conv6', conv6._keras_shape)

    after_conv3 = connection_block(conv3, 128, padding='same', activation=activation,
                                   pars=pars, allowed_pars=allowed_pars)
    print('after_conv3', after_conv3._keras_shape)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=3)
    conv7 = information_block(up7, 128, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv7', conv7._keras_shape)
    conv7 = Dropout(0.5)(conv7)
    print('conv7', conv7._keras_shape)

    after_conv2 = connection_block(conv2, 64, padding='same', activation=activation, pars=pars,
                                   allowed_pars=allowed_pars)
    print('after_conv2', after_conv2._keras_shape)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=3)
    conv8 = information_block(up8, 64, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv8', conv8._keras_shape)
    conv8 = Dropout(0.5)(conv8)
    print('conv8', conv8._keras_shape)

    after_conv1 = connection_block(conv1, 32, padding='same', activation=activation,
                                   pars=pars, allowed_pars=allowed_pars)
    print('after_conv1', after_conv1._keras_shape)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=3)
    conv9 = information_block(up9, 32, padding='same', activation=activation, pars=pars, allowed_pars=allowed_pars)
    print('conv9', conv9._keras_shape)
    conv9 = Dropout(0.5)(conv9)
    print('conv9', conv9._keras_shape)

    # main output
    conv10 = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid', name='main_output')(
        conv9)
    print('conv10', conv10._keras_shape)

    # creating a model
    # compiling the model
    if pars['outputs'] == 1:
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=optimizer,
                      loss={'main_output': dice_coef_loss},
                      metrics={'main_output': dice_coef})
    else:
        model = Model(inputs=inputs, outputs=[conv10, aux_out])
        model.compile(optimizer=optimizer,
                      loss={'main_output': dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                      metrics={'main_output': dice_coef, 'aux_output': 'acc'},
                      loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


# ----------------------------------------------------------------------------------------------------------------------

# get_unet() allows to try other versions of the u-net, if more are specified
get_unet = get_unet_customised

if __name__ == '__main__':
    # test the u-net without training

    img_rows = IMG_ROWS
    img_cols = IMG_COLS

    # to check that model works without training, any kind of optimiser can be used
    model = get_unet(Adam(lr=1e-5), pars=PARS)

    x = np.random.random((1, img_rows, img_cols, 1))
    result = model.predict(x, 1)
    print(result)
    print('params', model.count_params())
    print('layer num', len(model.layers))


########################################################################################################################
# ======================================================================================================================
# train
# ======================================================================================================================
########################################################################################################################

# standard-module imports
import numpy as np
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint, EarlyStopping

# # separate-module imports
# from u_model import get_unet, IMG_COLS as img_cols, IMG_ROWS as img_rows
# from data import load_train_data, load_test_data, load_nerve_presence
# from configuration import PARS, OPTIMIZER


def preprocess(imgs, to_rows=None, to_cols=None):
    """Resize all images in a 4D tensor of images of the shape (samples, rows, cols, channels).

    :param imgs: a 4D tensor of images of the shape (samples, rows, cols, channels)
    :param to_rows: new number of rows for images to be resized to
    :param to_cols: new number of rows for images to be resized to
    :return: a 4D tensor of images of the shape (samples, to_rows, to_cols, channels)
    """
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols

    print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], to_rows, to_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (to_rows, to_cols), preserve_range=True)
    return imgs_p


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_present = load_nerve_presence()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    # centering and standardising the images
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to be in {0, 1} instead of {0, 255}

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    # load model - the Learning rate scheduler choice is most important here
    model = get_unet(optimizer=OPTIMIZER, pars=PARS)

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(patience=5, verbose=1)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    if PARS['outputs'] == 1:
        imgs_labels = imgs_mask_train
    else:
        imgs_labels = [imgs_mask_train, imgs_present]

    model.fit(imgs_train, imgs_labels,
              batch_size=128, epochs=50,
              verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint, early_stopping])

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test = model.predict(imgs_test, verbose=1)

    if PARS['outputs'] == 1:
        np.save('imgs_mask_test.npy', imgs_mask_test)
    else:
        np.save('imgs_mask_test.npy', imgs_mask_test[0])
        np.save('imgs_mask_test_present.npy', imgs_mask_test[1])


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    train_and_predict()


########################################################################################################################
# ======================================================================================================================
# Submission
# ======================================================================================================================
########################################################################################################################

# standard-module imports
import os
import numpy as np
from skimage.transform import resize
from itertools import chain


# # separate-module imports
# from configuration import PARS
# from data import load_test_ids, image_rows, image_cols, _dir

def prep(img):
    """Prepare the image for to be used in a submission

    :param img: 2D image
    :return: resized version of an image
    """
    img = img.astype('float32')
    img = resize(img, (image_rows, image_cols), preserve_range=True)
    img = (img > 0.5).astype(np.uint8)  # threshold
    return img


def run_length_enc(label):
    """Create a run-length-encoding of an image

    :param label: image to be encoded
    :return: string with run-length-encoding of an image
    """
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]

    # consider empty all masks with less than 10 pixels being greater than 0
    if len(y) < 10:
        return ''

    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    """Create a submission .csv file.

    The file will have 2 cols: img, pixels.
        The image column consists of the ids of test images.
        The pixels column consists of the run-length-encodings of the corresponding images.
    """
    imgs_id_test = load_test_ids()

    print('Loading imgs_test from imgs_mask_test.npy')
    imgs_test = np.load('imgs_mask_test.npy')
    if PARS['outputs'] == 2:
        print('Loading imgs_exist_test from imgs_mask_test_present.npy')
        imgs_exist_test = np.load('imgs_mask_test_present.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    if PARS['outputs'] == 2:
        imgs_exist_test = imgs_exist_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []  # run-length-encodings
    for i in range(total):
        img = imgs_test[i, :, :, 0]
        if PARS['outputs'] == 2:
            img_exist = imgs_exist_test[i]
        img = prep(img)

        # only for version with 2 outputs
        if PARS['outputs'] == 2:
            # new probability of nerve presence
            new_prob = (img_exist + min(1, np.sum(img) / 10000.0) * 5 / 3) / 2
            # setting mask to array of zeros if new probability of nerve presence < 0.5
            if np.sum(img) > 0 and new_prob < 0.5:
                img = np.zeros((image_rows, image_cols))

        # producing run-length encoded version of the image
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    # creating a submission file
    file_name = os.path.join(_dir, 'submission.csv')
    with open(file_name, 'w+') as f:
        f.write('img,pixels\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


# --------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    submission()
