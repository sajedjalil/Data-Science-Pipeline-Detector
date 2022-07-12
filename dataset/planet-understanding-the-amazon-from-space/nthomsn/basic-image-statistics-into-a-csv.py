#=== DESCRIPTION ===
'''
This script prints a csv with a header and these variables:
image_name          - Filename of the image, same as in train.csv
bright_mean         - Average value of brightness for each pixel in the grayscale image
bright_stddev       - The standard deviation of this brightness
clear, road, etc    - T/F values for tags defined in TAGS config

Note: This takes minute or two for 40k images on my machine, change
DEMO to False in the config if you want to see more than the first 25
'''

#==== CONFIGURATION ====
# Change to False to use all images (takes a long time)
DEMO = True 
# Tags to create dummy (True/False) variables for
TAGS = ['haze','primary','agriculture','clear','water','habitation',
        'road','cultivation','cloudy','partly_cloudy']

TRAIN_CSV_FILE = '../input/train.csv'
TRAIN_JPG_PATH = '../input/train-jpg/'
TRUE = 'TRUE'
FALSE = 'FALSE'

#==== IMPORTS ====
from PIL import Image, ImageStat

#==== ACTUAL SCRIPT ====
# Convert image to grayscale, so there is one band
# Brightness is the average value of each pixel
def stats(img_file):
    img = Image.open(img_file).convert('L')
    stat = ImageStat.Stat(img)
    return (stat.mean, stat.stddev)

# Find features and return as csv string
def image_features(img_file):
    bright_mean, bright_stddev = stats(img_file)
    feature_str = ''
    feature_str += str(bright_mean[0]) + ','
    feature_str += str(bright_stddev[0])
    return feature_str

# Print header for all variables
print('image_name,bright_mean,bright_stddev,'+','.join(TAGS))

# Calculate features for all images, skipping first line
with open(TRAIN_CSV_FILE) as f:
    line_num = 0
    for line in f:
        line_num += 1
        if line_num == 1:
            continue
        if line_num > 10 and DEMO:
            break

        # Parse the train.csv given to us
        tokens = line.split(',')
        img_file = tokens[0]
        tags = tokens[1].strip().split(' ')

        # Print the image file, features, and a binary variable is_clear
        observation = ''
        observation += img_file + ','
        observation += image_features(TRAIN_JPG_PATH+img_file+'.jpg')

        dummies = ''
        for i in range(len(TAGS)):
            if TAGS[i] in tags:
                dummies += TRUE
            else:
                dummies += FALSE
            # Don't print last comma
            if i < len(TAGS)-1:
                dummies += ','
        print(observation+','+dummies)