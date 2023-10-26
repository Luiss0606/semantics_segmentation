import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('/home/luishuingo/semantics_segmentation/models') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('unet_costume.tflite', 'wb') as f:
  f.write(tflite_model)