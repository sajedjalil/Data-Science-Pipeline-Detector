import numpy as np
import cv2

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.preprocessing.image import Iterator


# Set this to whatever
IMG_TARGET_ROWS = 80
IMG_TARGET_COLS = 96


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


class ImageDataGenerator(Iterator):
    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None):
        self.X = self._preprocess(X)
        self.y = self._preprocess(y)

        # Add others as needed
        self.probs = {
            'keep': 0.1,
            'elastic': 0.9
        }

        super(ImageDataGenerator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def _preprocess(self, imgs):
        # You can try other things here like denoising filter etc.
        return [cv2.resize(img, (IMG_TARGET_COLS, IMG_TARGET_ROWS), interpolation=cv2.INTER_CUBIC) for img in imgs]

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

            # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(shape=(current_batch_size, 1, IMG_TARGET_ROWS, IMG_TARGET_COLS), dtype=np.uint8)
        batch_y = np.zeros(shape=(current_batch_size, 1, IMG_TARGET_ROWS, IMG_TARGET_COLS), dtype=np.float32)
        for i, j in enumerate(index_array):
            batch_x[i, 0], batch_y[i, 0] = self.apply_transform(self.X[j], self.y[j])
        return batch_x, batch_y

    def apply_transform(self, image, mask):
        prob_value = np.random.uniform(0, 1)
        if prob_value > self.probs['keep']:
            # You can add your own logic here.
            sigma = np.random.uniform(IMG_TARGET_COLS * 0.11, IMG_TARGET_COLS * 0.18)
            image = elastic_transform(image, IMG_TARGET_COLS, sigma)
            mask = elastic_transform(mask, IMG_TARGET_COLS, sigma)

        # Add other transforms here as needed. It will cycle through available transforms with give probs

        mask = mask.astype('float32') / 255.
        return image, mask
