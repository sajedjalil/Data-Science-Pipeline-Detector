import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten, Activation
from keras.layers.merge import Concatenate, add
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator



def build_model( baseline_cnn = False ):
    #Based on kernel https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d
    image_input = Input( shape = (75, 75, 3), name = 'images' )
    angle_input = Input( shape = [1], name = 'angle' )
    activation = 'elu'
    bn_momentum = 0.99
    
    # Simple CNN as baseline model
    if baseline_cnn:
        model = Sequential()

        model.add( Conv2D(16, kernel_size = (3, 3), activation = 'relu', input_shape = (75, 75, 3)) )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (3, 3), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Conv2D(32, kernel_size = (3, 3), activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (2, 2), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Conv2D(64, kernel_size = (3, 3), activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (2, 2), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Conv2D(128, kernel_size = (3, 3), activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( MaxPooling2D(pool_size = (2, 2), strides = (2, 2)) )
        model.add( Dropout(0.2) )

        model.add( Flatten() )

        model.add( Dense(256, activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( Dropout(0.3) )

        model.add( Dense(128, activation = 'relu') )
        model.add( BatchNormalization(momentum = bn_momentum) )
        model.add( Dropout(0.3) )

        model.add( Dense(1, activation = 'sigmoid') )

        opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

        model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

        model.summary()

    else:
        img_1 = Conv2D( 32, kernel_size = (3, 3), activation = activation, padding = 'same' ) ((BatchNormalization(momentum=bn_momentum) ) ( image_input) )
        img_1 = MaxPooling2D( (2,2)) (img_1 )
        img_1 = Dropout( 0.2 )( img_1 )

        img_1 = Conv2D( 64, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_1) )
        img_1 = MaxPooling2D( (2,2) ) ( img_1 )
        img_1 = Dropout( 0.2 )( img_1 )
  
         # Residual block
        img_2 = Conv2D( 128, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_1) )
        img_2 = Dropout(0.2) ( img_2 )
        img_2 = Conv2D( 64, kernel_size = (3, 3), activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_2) )
        img_2 = Dropout(0.2) ( img_2 )
        
        img_res = add( [img_1, img_2] )

        # Filter resudial output
        img_res = Conv2D( 128, kernel_size = (3, 3), activation = activation ) ( (BatchNormalization(momentum=bn_momentum)) (img_res) )
        img_res = MaxPooling2D( (2,2) ) ( img_res )
        img_res = Dropout( 0.2 )( img_res )
        img_res = GlobalMaxPooling2D() ( img_res )
        
        cnn_out = ( Concatenate()( [img_res, BatchNormalization(momentum=bn_momentum)(angle_input)]) )

        dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (Dense(256, activation = activation) (cnn_out)) )
        dense_layer = Dropout( 0.5 ) ( BatchNormalization(momentum=bn_momentum) (Dense(64, activation = activation) (dense_layer)) )
        output = Dense( 1, activation = 'sigmoid' ) ( dense_layer )
        
        model = Model( [image_input, angle_input], output )

        opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

        model.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'] )

        model.summary()

    return model

def get_callbacks( weight_save_path, no_improv_epochs = 10, min_delta = 1e-4 ):
    es = EarlyStopping( 'val_loss', patience = no_improv_epochs, mode = 'min', min_delta = min_delta )
    ms = ModelCheckpoint( weight_save_path, 'val_loss', save_best_only = True )

    return [ es, ms ]

def generate_data( data ):
    X_band_1=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_1']] )
    X_band_2=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_2']] )
    X = np.concatenate( [X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], \
                        ((X_band_1 + X_band_2)/2)[:, :, :, np.newaxis]], axis=-1 )
    return X

def augment_data( generator, X1, X2, y, batch_size = 32 ):
    generator_seed = np.random.randint( 9999 )
    gen_X1 = generator.flow( X1, y, batch_size = batch_size, seed = generator_seed )
    gen_X2 = generator.flow( X1, X2, batch_size = batch_size, seed = generator_seed )

    while True:
        X1i = gen_X1.next()
        X2i = gen_X2.next()

        yield [ X1i[0], X2i[1] ], X1i[1]
    
def plot_band_samples( data, band = 1, title = None ):
    fig = plt.figure( 1, figsize=(15, 15) )
    for i in range(9):
        ax = fig.add_subplot( 3, 3, i + 1 )
        arr = np.reshape( np.array(data.iloc[i, band - 1]), (75, 75) )
        ax.imshow( arr, cmap='inferno' )
        fig.suptitle( title )

    plt.show()

def plot_all_bands( data, title = None ):
    fig = plt.figure( 1, figsize = (15, 15) )
    count = 1
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot( 3, 3, count )
            ax.imshow( data[i, :, :, j], cmap = 'inferno' )
            count += 1
            if i == 0:
                if j == 0:
                    ax.set_title( 'Band 1' , fontsize = 12)
                elif j == 1:
                    ax.set_title( 'Band 2', fontsize = 12 )
                elif j == 2:
                    ax.set_title( 'Average', fontsize = 12 )
    fig.suptitle( title, fontsize = 14, fontweight = 'bold' )
    plt.show()

def make_plots( data, band_samples = True, all_bands = True ):
    ships = data[ data.is_iceberg == 0 ].sample( n = 9, random_state = 42 )
    icebergs = data[ data.is_iceberg == 1 ].sample( n = 9, random_state = 42 )

    np_ships = generate_data( ships )
    np_icebergs = generate_data( icebergs )

    if band_samples:
        plot_band_samples( ships, band = 2, title = 'Ship image samples' )
        plot_band_samples( icebergs, band = 2, title = 'Iceberg image samples' )

    if all_bands:
        plot_all_bands( np_ships, 'Image bands for ships' )
        plot_all_bands( np_icebergs, 'Image bands for icebergs' )

 
 
 
TEST = True # Should test data be passed to the model?
DO_PLOT = False # Exploratory data plots
USE_AUGMENTATION = True # Whether or not image augmentations should be made
TRAIN_PATH = '../input/train.json'
TEST_PATH = '../input/test.json'
WEIGHT_SAVE_PATH = 'model_weights.hdf5'
PREDICTION_SAVE_PATH = 'test_submission.csv'

if TEST:
    SEED = np.random.randint( 9999 )
else:
    SEED = 42 # Constant seed for comparability between runs

BATCH_SIZE = 32
EPOCHS = 10 # Increase this

train_data = pd.read_json( TRAIN_PATH )
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].replace('na', 0)
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].astype(float).fillna(0.0)

X = generate_data( train_data )
X_a = train_data[ 'inc_angle' ]
y = train_data[ 'is_iceberg' ]

if DO_PLOT:
    make_plots( train_data, band_samples = True, all_bands = True )

X_train, X_val, X_angle_train, X_angle_val, y_train, y_val = train_test_split( X, X_a, y, train_size = .8, random_state = SEED )
callback_list = get_callbacks( WEIGHT_SAVE_PATH, 20 )

model = build_model()
start_time = time.time()

if USE_AUGMENTATION:
    image_augmentation = ImageDataGenerator( rotation_range = 20,
                                             horizontal_flip = True,
                                             vertical_flip = True,
                                             width_shift_range = .3,
                                             height_shift_range =.3,
                                             zoom_range = .1 )
    input_generator = augment_data( image_augmentation, X_train, X_angle_train, y_train, batch_size = BATCH_SIZE )

    model.fit_generator( input_generator, steps_per_epoch = 4096/BATCH_SIZE, epochs = EPOCHS,
                         callbacks = callback_list, verbose = 2, 
                         validation_data = augment_data(image_augmentation, X_val, X_angle_val, y_val, batch_size = BATCH_SIZE),
                         validation_steps = len(X_val)/BATCH_SIZE )

else: 
    # Just fit model to the given training data
    model.fit( [X_train, X_angle_train], y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 2, 
               validation_data = ([X_val, X_angle_val], y_val), callbacks = callback_list )

m, s = divmod( time.time() - start_time, 60 )
print( 'Model fitting done. Total time: {}m {}s'.format(int(m), int(s)) )

model.load_weights( WEIGHT_SAVE_PATH )
val_score = model.evaluate( [X_val, X_angle_val], y_val, verbose = 1 )
print( 'Validation score: {}'.format(round(val_score[0], 5)) )
print( 'Validation accuracy: {}%'.format(round(val_score[1]*100, 2)) )
print( '='*20, '\n' )

if TEST:
    print( 'Loading and evaluating on test data' )
    test_data = pd.read_json( TEST_PATH )

    X_test = generate_data( test_data )
    X_a_test = test_data[ 'inc_angle' ]
    test_predictions = model.predict( [X_test, X_a_test] )

    submission = pd.DataFrame()
    submission[ 'id' ] = test_data[ 'id' ]
    submission[ 'is_iceberg' ] = test_predictions.reshape( (test_predictions.shape[0]) )

    submission.to_csv( PREDICTION_SAVE_PATH, index = False )