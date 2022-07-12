import tensorflow as tf
import os
from zipfile import ZipFile

keras_model_file = "../input/google-landmark-retrieval-2020-keras-test-model/model.keras"

target_size = (256,256)
keep_aspect_ratio = False
normalize = True

# Load model with input shape [None,target_size,target_size,3] and output [None,D]
encoder_model = tf.keras.models.load_model(keras_model_file)

class EncasedModelModule(tf.Module):
    def __init__(self):
      super(EncasedModelModule, self).__init__()
      # Create model instance
      self.encoder = encoder_model
      
    @tf.function (input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')])
    def serve(self, input_image):
      # Add batch dimension
      x = tf.expand_dims(input_image, 0, name='input_image')
      # Resize image
      x = tf.image.resize(x, target_size, method=tf.image.ResizeMethod.LANCZOS5 , preserve_aspect_ratio=keep_aspect_ratio, antialias=True, name=None)
      # Normalize
      if normalize: x = tf.image.per_image_standardization(x)
      # Run processed image through model
      y = self.encoder(x)
      # Reduce output to single dimension
      y = tf.squeeze(y, axis=0) #, name='encoding_flat')
      named_output_tensors = {'global_descriptor' : tf.identity(y, name='global_descriptor')}
      return named_output_tensors

module = EncasedModelModule()

# Save encased model as tensorflow pb file
tf.saved_model.save(module, './model', signatures={"serving_default" : module.serve})

# Zip model for submit
with ZipFile('submission.zip','w') as output_zip_file:
    for filename in os.listdir('./model'):
      if os.path.isfile('./model/'+filename):
         output_zip_file.write('./model/'+filename, arcname=filename) 
    for filename in os.listdir('./model/variables'):
      if os.path.isfile('./model/variables/'+filename):
         output_zip_file.write('./model/variables/'+filename, arcname='variables/'+filename)
    for filename in os.listdir('./model/assets'):
      if os.path.isfile('./model/assets/'+filename):
         output_zip_file.write('./model/assets/'+filename, arcname='assets/'+filename)

print('Model succesfully saved.')

# Select the "submission.zip" file in the output list as submission