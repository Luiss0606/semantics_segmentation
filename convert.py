from json import load
import tensorflow as tf
from tensorflow.keras.models import load_model

# Convert the model
model_keras = load_model('unet_personal_v2.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
tflite_model = converter.convert()

# Save the model.
with open('unet_personal_v2.tflite', 'wb') as f:
  f.write(tflite_model)