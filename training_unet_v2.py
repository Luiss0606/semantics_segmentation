import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


def create_unet_model(input_size=(128, 128, 3), num_classes=1, segmentation_type='binary'):
    # Base preentrenada de MobileNetV2
    base_model = MobileNetV2(input_shape=input_size, include_top=False)
    
    # Capas de interés del modelo base para la conexión
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Crear el modelo de extracción de características
    down_stack = Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = True

    # Definición de la parte ascendente del U-Net
    up_stack = [
        UpSampling2D((2, 2)),  # 4x4 -> 8x8
        UpSampling2D((2, 2)),  # 8x8 -> 16x16
        UpSampling2D((2, 2)),  # 16x16 -> 32x32
        UpSampling2D((2, 2)),  # 32x32 -> 64x64
        UpSampling2D((2, 2)),  # 64x64 -> 128x128
    ]

    inputs = Input(shape=input_size)
    x = inputs

    # Descenso en el U-Net
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Ascenso en el U-Net y concatenación con las capas de salto
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

    # Capa final de convolución dependiendo del tipo de segmentación
    if segmentation_type == 'binary':
        x = Conv2D(1, 3, activation='sigmoid')(x)
    else:
        x = Conv2D(num_classes, 3, activation='softmax')(x)

    # Asegurarse de que la salida sea de 128x128
    x = tf.image.resize(x, (128, 128))

    model = Model(inputs=inputs, outputs=x)
    loss = 'binary_crossentropy' if segmentation_type == 'binary' else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model

# Configuración del modelo
model = create_unet_model(segmentation_type='binary')



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = '/home/luishuingo/semantics_segmentation/Floor-Segmentationv1/train'
TEST_PATH = '/home/luishuingo/semantics_segmentation/Floor-Segmentationv1/test'
VALID_PATH = '/home/luishuingo/semantics_segmentation/Floor-Segmentationv1/valid/'

X_train = np.zeros((len(os.listdir(TRAIN_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(os.listdir(TRAIN_PATH)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

print('Resizing training images and masks')

for n, id_ in enumerate(os.listdir(TRAIN_PATH)):
    
    # Load and resize the training image
    img = cv2.imread(os.path.join(TRAIN_PATH, id_))  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)  # Resize the image
    X_train[n] = img

    # Load and resize the corresponding mask
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    mask_ = cv2.imread(os.path.join(TRAIN_PATH, id_.replace('.jpg', '_mask.png')), cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
    mask_ = cv2.resize(mask_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)  # Resize the mask
    mask_ = np.expand_dims(mask_, axis=-1)
    mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Now we do the same for the test set
X_test = np.zeros((len(os.listdir(TEST_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(os.listdir(TEST_PATH)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

print('Resizing test images and masks')

for n, id_ in enumerate(os.listdir(TEST_PATH)):

    # Load and resize the training image
    img = cv2.imread(os.path.join(TEST_PATH, id_))  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)  # Resize the image
    X_test[n] = img

    # Load and resize the corresponding mask
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    mask_ = cv2.imread(os.path.join(TEST_PATH, id_.replace('.jpg', '_mask.png')), cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
    mask_ = cv2.resize(mask_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)  # Resize the mask
    mask_ = np.expand_dims(mask_, axis=-1)
    mask = np.maximum(mask, mask_)
    Y_test[n] = mask

# Now we do the same for the validation set

X_valid = np.zeros((len(os.listdir(VALID_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_valid = np.zeros((len(os.listdir(VALID_PATH)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

print('Resizing validation images and masks')


# Model checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
              tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir='logs')]

model.summary()
results = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=16, epochs=100, callbacks=callbacks,)

# Save the modelin tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open('unet_personal_v2.1_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)