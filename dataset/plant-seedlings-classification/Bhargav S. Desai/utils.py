"""
This is a utility script with a couple of helper classes to make the overall structure cleaner.

DataLoader ->  Builds a clean interface to handle all the preprocessing, balancing and loading of the dataset for visualization or training.
Utilities -> Allows easy plotting of graphs and summarizing models after training them.

"""


import numpy as np
import itertools
import pandas as pd
import os
import math
import random
import cv2
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.activations import swish
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from imgaug import augmenters as iaa
from tqdm.notebook import tqdm





class DataLoader():
    """
    Args:
    train_dir -- points to the training directory
    test_dir -- points to the test directory
    save_dir -- points to the path where augmented data should be stored
    segmentation -- determines whether to apply segmentation during preprocessing or not
    target_size -- the size to which each image in the dataset should be resized
    
    Returns:
    An instance of itself
    """
    
    def __init__(self, **kwargs):
        
        self.train_dir = kwargs.get('train_dir')
        self.test_dir = kwargs.get('test_dir')
        self.save_dir = kwargs.get('save_dir')
        self.segmentation = kwargs.get('segmentation')
        self.target_size = kwargs.get('target_size')
        categories = [category for category in sorted(os.listdir(self.train_dir))]
        
            
        
    # Helper Function 1
    # Create a binary mask for a given HSV range
    def create_mask_for_plant(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([25, 50, 50])
        upper_hsv = np.array([95, 255, 255])
        mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # Helper Function 2
    # Use the binary mask to segment the image
    def segment_plant(self, image):
        mask = self.create_mask_for_plant(image)
        output = cv2.bitwise_and(image, image, mask = mask)
        return output

    # Helper Function 3
    # Sharpen the segmented image for contrast
    def sharpen_image(self, image):
        #image_median_blurred = cv2.medianBlur(image, 3)
        image_sharp = cv2.bilateralFilter(image, 11, 11, 11) 
        #image_blurred = cv2.GaussianBlur(image_median_blurred, (0, 0), 3)
        #image_sharp = cv2.addWeighted(image, 1.5, image_median_blurred, -0.5, 0)
        return image_sharp

    # Helper Function 4 
    # Augment passed images
    def augment_images(self, class_images):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-45, 45)),
            iaa.TranslateX(percent=(-0.1, 0.1)),
            iaa.TranslateY(percent=(-0.1, 0.1))
        ], random_order=True)

        images_aug = seq(images = class_images)
        return images_aug

    # Helper Function 5
    # Randomly sample images from a set of passed images
    def random_unique_sampling(self, class_images, remainder):
        random_unique_indices = random.sample(range(0, len(class_images)), remainder)
        random_unique_images = [class_images[idx] for idx in random_unique_indices]
        return random_unique_images
        

    def augmentation_pipeline(self, class_images, number_of_images):
        """Accepts a batch of images (of a single class) and returns a required number of augmented images"""

        if number_of_images == 0:
                return []

        elif number_of_images >= len(class_images):
            batches = math.floor(number_of_images / len(class_images))
            remainder = number_of_images % len(class_images)
            remainder_images = self.random_unique_sampling(class_images, remainder)
            class_images = class_images * batches
            class_images.extend(remainder_images)
            images_aug = self.augment_images(class_images)
            return images_aug

        else:
            assert number_of_images < len(class_images)
            class_images = self.random_unique_sampling(class_images, number_of_images)
            images_aug = self.augment_images(class_images)
            return images_aug
        
    def preprocessing_pipeline(self, path):
        """Accepts a path and returns a processed image involving reading and resizing"""
        image = cv2.resize(cv2.imread(path), self.target_size, interpolation = cv2.INTER_NEAREST)
        return image


    def segmentation_pipeline(self, image):
        """Accepts an image and returns a HSV segmented version of the image"""
        image_segmented = self.segment_plant(image)
        image_sharpen = self.sharpen_image(image_segmented)
        return image_sharpen
    
    
    def balance_dataset(self):
        """Create augmented data to balance classes from the passed training data path"""
        
        # Make a directory for augmented dataset
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Get categories
        categories = [category for category in sorted(os.listdir(self.train_dir))]

        # Get the maximum amount of images that exists in a class
        max_in_class = max([len(os.listdir(os.path.join(self.train_dir, category))) for category in categories])

        # Find out the augmentations required for each class
        images_per_category = {category : len(os.listdir(os.path.join(self.train_dir, category))) for category in categories}

        # Find out the augmentations required for each class
        required_augmentations = dict(zip(categories,  [max_in_class - num_in_class for num_in_class in list(images_per_category.values())]))

        # Augment each unbalanced class and save the new dataset to disk
        # We preferring saving the data to disk
        # Because we prefer to not hold large numpy arrays in the RAM
        # This allows for large models to be loaded and trained on
        # We use for loops here instead of list comprehensions for readiblity
        for category in tqdm(categories):
            try:
                os.mkdir(os.path.join(self.save_dir, category))
            except FileExistsError:
                pass
            class_images = list()

            # Preprocessing and Augmentation
            for img_path in sorted(os.listdir(os.path.join(self.train_dir, category))):
                image = self.preprocessing_pipeline(os.path.join(self.train_dir, category, img_path))
                if self.segmentation == True:
                    image = self.segmentation_pipeline(image)
                class_images.append(image)
            augmented_images = self.augmentation_pipeline(class_images, required_augmentations[category])
            class_images.extend(augmented_images)

            # Writing the augmented data to disk
            for image_number, class_image in enumerate(class_images):
                cv2.imwrite(os.path.join(self.save_dir, category, "{}.png".format(image_number + 1)), class_image)
        
    def load_for_train(self, model):
        
        if model == "resnet50":
            datagen = ImageDataGenerator(preprocessing_function = resnet50.preprocess_input, validation_split=0.15)
            target_size = (224, 224)
        elif model == "inception_v3":
            datagen = ImageDataGenerator(preprocessing_function = inception_v3.preprocess_input, validation_split=0.15)
            target_size = (299, 299)
        elif model == 'vgg16':
            datagen = ImageDataGenerator(preprocessing_function = vgg16.preprocess_input, validation_split=0.15)
            target_size = (224, 224)
        else:
            sys.exit('Fatal Error: Invalid Model Requested.')


        train_generator = datagen.flow_from_directory(
                directory= os.path.join(self.save_dir),
                target_size= self.target_size,
                class_mode = "categorical",
                batch_size=32,
                shuffle=True,
                subset='training'
            )
        
        val_generator = datagen.flow_from_directory(
                directory= os.path.join(self.save_dir),
                target_size= self.target_size,
                class_mode = 'categorical',
                batch_size=32,
                shuffle=False,
                subset='validation'
            )

        return train_generator, val_generator
        
    def load_for_viz(self, model):
        
        if model == "resnet50":
            datagen = ImageDataGenerator(preprocessing_function = resnet50.preprocess_input, validation_split=0.15)
            target_size = (224, 224)
        elif model == "inception_v3":
            datagen = ImageDataGenerator(preprocessing_function = inception_v3.preprocess_input, validation_split=0.15)
            target_size = (299, 299)
        elif model == 'vgg16':
            datagen = ImageDataGenerator(preprocessing_function = vgg16.preprocess_input, validation_split=0.15)
            target_size = (224, 224)
        else:
            sys.exit('Fatal Error: Invalid Model Requested.')

        generator = datagen.flow_from_directory(
        directory= os.path.join(self.save_dir),
        target_size= self.target_size,
        batch_size=1,
        class_mode=None,
        shuffle=False
        )
        
        categories = [category for category in sorted(os.listdir(self.train_dir))]
        max_in_class = max([len(os.listdir(os.path.join(self.train_dir, category))) for category in categories])
        categories_rep = list(itertools.chain.from_iterable(itertools.repeat(x, max_in_class) for x in categories))
        data_df = pd.DataFrame(categories_rep, columns = ["categories"])
        
        return generator, data_df
    
    def load_for_inference(self):
        
        test_images = np.array([self.segmentation_pipeline(self.preprocessing_pipeline(os.path.join(self.test_dir, img_path))) for img_path in sorted(os.listdir(self.test_dir))])
        filenames = [filename for filename in sorted(os.listdir(self.test_dir))]
        return test_images, filenames
        
    
    def show_sample_images(self):
        categories = [category for category in sorted(os.listdir(self.train_dir))]
        random_indices = random.sample(range(0, len(self.data_og)), 4)
        
        # Plot some sample images from the dataset
        _, axs = plt.subplots(1, 4, figsize=(20, 20))
        for i in range(4):
            axs[i].imshow(self.data_og[random_indices[i]])
        
        # Plot segmented images if segmentation is True
        if self.segmentation:
            _, axs = plt.subplots(1, 4, figsize=(20, 20))
            for i in range(4):
                axs[i].imshow(self.data_seg[random_indices[i]])
            
            
            
class Utilities:
    """
    Boilerplate code that can be re-used multiple times to plot training graphs, visualization plots, training summary.
    
    Args:
    train_dir -- points to the training directory
    save_dir -- points to the save directory
    
    Returns:
    An instance of itself
    """
    
    def __init__(self, train_dir, save_dir):
        self.train_dir = train_dir
        self.save_dir = save_dir
        
        
    def plot_tSNE(self, data_df, base_model, generator, title):
        
        feature_vector = base_model.predict_generator(generator, 7848, verbose =1)
        print('Extratced feature dimensionality: {}'.format(feature_vector.shape))
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
        tsne_results = tsne.fit_transform(feature_vector)
        
        data_df['tsne-2d-one'] = tsne_results[:,0]
        data_df['tsne-2d-two'] = tsne_results[:,1]

        plt.figure(figsize=(20,16))
        plt.title("tSNE Visualization - " + title)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="categories",
            palette=sns.color_palette("hls", 12),
            data=data_df,
            legend="full",
            alpha=0.3
        )
        plt.savefig(os.path.join(os.path.normpath(self.save_dir + os.sep + os.pardir), title + ".png"), dip=300, bbox_inches='tight')
        
    def summarize_model(self, history_model, model, val_generator):
        categories = [category for category in sorted(os.listdir(self.train_dir))]
        self.plot_curves(history_model)
        self.plot_classification_metrics(categories, model, val_generator)


    def plot_curves(self, history_model):
        plt.style.use('seaborn')

        # Summarize history for accuracy
        plt.figure(1, figsize=(16, 10))
        plt.plot(history_model.history['accuracy'])
        plt.plot(history_model.history['val_accuracy'])
        plt.title('Train and Validation Accuracy', fontsize = 16)
        plt.ylabel('Accuracy', fontsize = 14)
        plt.xlabel('Epoch', fontsize = 14)
        plt.legend(['Train', 'Validation'], fontsize = 14)
        plt.show()

        # Summarize history for loss
        plt.figure(2, figsize=(16, 10))
        plt.plot(history_model.history['loss'])
        plt.plot(history_model.history['val_loss'])
        plt.title('Train and Validation Loss', fontsize = 16)
        plt.ylabel('Loss', fontsize = 14)
        plt.xlabel('Epoch', fontsize = 14)
        plt.legend(['Train', 'Validation'], fontsize = 14)
        plt.show()

    def plot_classification_metrics(self, categories, model, val_generator):

        predictiions = model.predict_generator(val_generator, 48)
        y_pred = np.argmax(predictiions, axis=1)
        cf_matrix = confusion_matrix(val_generator.classes, y_pred)
        print('Classification Report')
        print(classification_report(val_generator.classes, y_pred, target_names=categories))
        plt.figure(figsize=(20,20))
        sns.heatmap(cf_matrix, annot=True, xticklabels=categories, yticklabels=categories, cmap='Blues')
        
        
    def infer(self, test_images, model):
        categories = [category for category in sorted(os.listdir(self.train_dir))]
        predictions = model.predict(test_images, batch_size = 32)
        y_pred = np.argmax(predictions, axis = 1)
        y_pred_categories = [categories[i] for i in y_pred]
        return y_pred_categories
        
        
    def make_csv(self, y_pred_categories, filenames, save_path):
        inference = pd.DataFrame(zip(filenames, y_pred_categories), columns = ['Filename', "Prediction"])
        inference.to_csv(os.path.join(save_path, "inference.csv"))
        print('Saved inferences to disk!')