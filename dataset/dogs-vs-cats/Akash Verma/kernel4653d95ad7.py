import os
import cv2
import numpy as np
import pandas as pd
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

filenames = os.listdir("../input/train/train")
test_files = os.listdir("../input/test1/test1")

# Creating a target list which will contain target labels
target = []
for name in filenames:
    if "cat" in name:
        target.append('cat')
    else:
        target.append('dog')
train = pd.DataFrame({'filenames' : filenames,
                     'target' : target})

# test dataset contains only filenames 
test =  pd.DataFrame({'filenames' : test_files})

# Splitting data for training and validation
x_train, x_test, y_train, y_test = train_test_split(train['filenames'], 
                                                    train['target'], 
                                                    test_size=0.2,
                                                    stratify=train['target'])

# Again creating the dataframe of splitted data
training_dataset = pd.DataFrame({'filename' : x_train, 
                       'target' : y_train})
validation_dataset = pd.DataFrame({'filename' : x_test, 
                       'target' : y_test}) 
BATCH_SIZE = 20
train_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)
test_generator = ImageDataGenerator(rescale=1/.255)

train_gen = train_generator.flow_from_dataframe(training_dataset, 
                                                "../input/train/train/",
                                                target_size = (150, 150),
                                                batch_size = 20,
                                                x_col = 'filename',
                                                y_col = 'target',
                                                class_mode = 'binary',
                                                shuffle = False)

validation_gen = test_generator.flow_from_dataframe(validation_dataset, 
                                                    "../input/train/train/",
                                                    target_size = (150, 150),
                                                    batch_size = 20,
                                                    x_col = 'filename',
                                                    y_col = 'target',
                                                    class_mode = 'binary',
                                                    shuffle = False)

test_gen = test_generator.flow_from_dataframe(test,
                                              "../input/test1/test1/",
                                              target_size = (150, 150),
                                              batch_size = 20,
                                              x_col = 'filenames',
                                              class_mode = None,
                                              shuffle = False)
for data_batch, label_batch in train_gen:
    print("Data Shape ", data_batch.shape)
    print('Target Shape ',label_batch.shape)
    plt.imshow(data_batch[4])
    break
model = Sequential()
# Layer 1
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

#Layer 2
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

#Layer 3
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

#Layer 4
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

# Adding a flatten layer
model.add(layers.Flatten())

# Addling a dense layer
model.add(layers.Dense(512, activation='relu'))
# Final output layer
model.add(layers.Dense(1, activation = 'sigmoid'))

# Compling the model
model.compile(optimizer = optimizers.RMSprop(lr = 1e-4),
             loss='binary_crossentropy',
             metrics=['acc'])
history = model.fit_generator(
    train_gen, 
    epochs=2,
    validation_data=validation_gen,
    validation_steps=250, # test_dataset.shape[0]//20,
    steps_per_epoch=100 # training_dataset.shape[0]//20
)
model.save('catsDogsClassification.h5')
print(model.summary())
predicted_value = np.zeros(validation_dataset.shape[0])
i = 0
df = pd.DataFrame(columns = ['predicted_values', 
                             'original_values'])
for test_data, test_label in validation_gen:
    predicted_value = model.predict(test_data)
    predicted_value = np.reshape(predicted_value, (20,)).tolist()
    predicted_df = pd.DataFrame({'predicted_values' : predicted_value,
                                'test_label' : test_label})
    df = pd.concat([df, predicted_df],
                   axis = 0,
                   ignore_index=False,
                   sort = False)
    i = i + 1
    if i*20 >= validation_dataset.shape[0]:
        break
        
# Predicting on the validation dataset
predicted_df['predicted_values'] = predicted_df['predicted_values'].apply(lambda x : 1 if x > 0.5 else 0)

class_mapping = train_gen.class_indices
print('F1 Score on the validation data set - ', f1_score(predicted_df['predicted_values'] , 
                                                      predicted_df['test_label']))
test_gen.reset()
prediction =  model.predict_generator(test_gen,
                                     steps=test.shape[0]/BATCH_SIZE)
submission_file = pd.DataFrame({'id' : test_gen.filenames,
                               'label' : np.reshape(prediction, (prediction.shape[0], )).astype(int)})
submission_file['id'] = submission_file['id'].str.replace('.jpg', '')

submission_file.to_csv('submission.csv', index = False)