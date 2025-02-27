import os
import tensorflow as tf
base_dir = os.getcwd()
train_data_dir = os.path.join(base_dir , 'Products', 'train')
test_data_dir = os.path.join(base_dir , 'Products', 'test')

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
baseModel = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SHAPE
)

baseModel.trainable = False

model = tf.keras.Sequential(
    [
        baseModel,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation="softmax")
    ]
)

trainDatagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
   horizontal_flip=True,
   width_shift_range=0.2,
   height_shift_range=0.2,
   rotation_range=15,
   vertical_flip=True,
   fill_mode='reflect',
   data_format='channels_last',
   brightness_range=[0.5, 1.5])

testDatagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

train_generator = trainDatagen.flow_from_directory(
        train_data_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE
        )

test_generator = testDatagen.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE
    )

history = model.fit(train_generator, epochs=20, validation_data=test_generator)

layer_outputs = [layer.output for layer in model.layers[0:]]
layer_outputs

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save(base_dir+'SchneiderProducts.h5')