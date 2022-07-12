import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from keras.layers.merge import Concatenate, add
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

SEED       = 816  # Random seed
TRAIN_PATH = '../input/train.json'
TEST_PATH  = '../input/test.json'

BATCH_SIZE     = 32
EPOCHS         = 15 
EARLY_STOPPING = 7  # If model loss doesn't improve after this many iterations then stop

def build_model( ):
    
    image_input = Input( shape = (75, 75, 3), name = 'images' )
    angle_input = Input( shape = [1], name = 'angle' )
    activation = 'relu'
    bn_momentum = 0.99
    
    img_1 = BatchNormalization(momentum=bn_momentum)( image_input )
    img_1 = Conv2D( 32, kernel_size = (3, 3), activation = activation, padding = 'same' )(img_1)

    img_1 = MaxPooling2D( (2,2)) (img_1 )
    
    img_1 = Conv2D( 64, kernel_size = (3, 3), activation = activation, 
                    padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_1) )
    img_1 = MaxPooling2D( (2,2), name='skip1' ) ( img_1 )
    
     # Residual block
    img_2 = Conv2D( 128, kernel_size = (3, 3), activation = activation, 
                    padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_1))
    img_2 = Conv2D( 64, name='img2', kernel_size = (3, 3), 
                    activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_2) )
    
    img_2 = add( [img_1, img_2] )
    img_2 = MaxPooling2D( (2,2), name='skip2' ) ( img_2 )
    
    # Residual block
    img_3 = Conv2D( 128, kernel_size = (3, 3), activation = activation, 
                    padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_2))
    img_3 = Conv2D( 64, name='img3', kernel_size = (3, 3), 
                    activation = activation, padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_3))
    
    img_res = add( [img_2, img_3] )

    # Filter residual output
    img_res = Conv2D( 128, kernel_size = (3, 3), 
                      activation = activation ) ((BatchNormalization(momentum=bn_momentum)) (img_res))
    
    # Can you guess why we do this? Hint: Where did Flatten go??
    img_res = GlobalMaxPooling2D(name='global_pooling') ( img_res )
    
    # What is this? Hint: We have 2 inputs. An image and a number.
    cnn_out = Concatenate(name='What_happens_here')( [img_res, angle_input] )

    dense_layer = Dropout( 0.5 ) (Dense(256, activation = activation) (cnn_out)) 
    dense_layer = Dropout( 0.5 )  (Dense(64, activation = activation) (dense_layer)) 
    output = Dense( 1, activation = 'sigmoid' ) ( dense_layer )
    
    model = Model( [image_input, angle_input], output )

    opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

    model.compile( loss = 'binary_crossentropy', 
                   optimizer = opt, 
                   metrics = ['accuracy'] )

    model.summary()

    return model


def get_callbacks( no_improv_epochs = 10, min_delta = 1e-4 ):
    
    # Early stopping - End training early if we don't improve on the loss function
    # by a certain minimum threshold
    es = EarlyStopping( 'val_loss', patience = no_improv_epochs, 
                        mode = 'min', min_delta = min_delta )
    
    return [ es ]

# This generates the "image". It is really 2 spectral bands
# from a sattelite. So each band will be considered a "color" channel.
# They define channel 1 as band 1, channel 2 as band 2, and channel 3
# as band1+band2.
# Suman: Some people have done a similar 
# thing on the I/Q components of wireless signals.
def generate_data( data ):
    X_band_1=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_1']] )
    X_band_2=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_2']] )
    X = np.concatenate( [X_band_1[:, :, :, np.newaxis], 
                         X_band_2[:, :, :, np.newaxis], \
                        ((X_band_1 + X_band_2)/2)[:, :, :, np.newaxis]], axis=-1 )
    return X

# This creates a Python iterator over the inputs
# It will run continually, but yield one batch at a time.
# So whenever you make the call to 'augment_data'
# it will return the next batch of data for you
def augment_data( generator, X1, X2, y, batch_size = 32 ):
    generator_seed = np.random.randint( SEED )
    gen_X1 = generator.flow( X1, y, 
                             batch_size = batch_size, seed = generator_seed )
    gen_X2 = generator.flow( X1, X2, 
                             batch_size = batch_size, seed = generator_seed )

    while True:
        X1i = gen_X1.next()
        X2i = gen_X2.next()

        yield [ X1i[0], X2i[1] ], X1i[1]
    

# Read the angle data and replace NaN with 0 degrees
train_data = pd.read_json( TRAIN_PATH )
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].replace('na', 0)
train_data[ 'inc_angle' ] = train_data[ 'inc_angle' ].astype(float).fillna(0.0)

# The data images are actually the spectral bands of a satellite
X = generate_data( train_data )
X_a = train_data[ 'inc_angle' ]
y = train_data[ 'is_iceberg' ]

# Divide into test/train sets
X_train, X_val, X_angle_train, X_angle_val, y_train, y_val = \
    train_test_split( X, X_a, y, 
                      train_size = .8, 
                      random_state = SEED )

# Define the ResNet model topology
model = build_model()

# Our data augmentation (applied to just the image)
image_augmentation = ImageDataGenerator( rotation_range = 20,
                                         horizontal_flip = True,
                                         vertical_flip = True,
                                         width_shift_range = .3,
                                         height_shift_range =.3,
                                         zoom_range = .1 )
train_generator = augment_data( image_augmentation, X_train, 
                                X_angle_train, y_train, 
                                batch_size = BATCH_SIZE )

# Stop after a certain number of epochs if no benefit
callback_list = get_callbacks(EARLY_STOPPING)

# Perform SGD
model.fit_generator( train_generator, 
                     steps_per_epoch = len(X_train)//BATCH_SIZE, 
                     epochs = EPOCHS,
                     callbacks = callback_list, verbose = 1, 
                     validation_data = [[X_val, X_angle_val], y_val] )


# Print the accuracy on the test set
val_score = model.evaluate( [X_val, X_angle_val], y_val, verbose = 1 )
print( 'Validation score: {:.3f}'.format(val_score[0]) )
print( 'Validation accuracy: {:.2f}%'.format(val_score[1]*100) )
print( '='*20, '\n' )