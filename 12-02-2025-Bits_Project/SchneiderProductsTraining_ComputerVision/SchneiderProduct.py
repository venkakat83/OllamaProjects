import os
import tensorflow as tf

# Test and Train Data of the schneider prouducts
base_dir = os.getcwd()
train_data_dir = os.path.join(base_dir , 'Products', 'train')
test_data_dir = os.path.join(base_dir , 'Products', 'test')

# Set Image Width and Image Height
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

def get_base_model_instance(weights, image_shape):
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                   include_top=False,
                                                   weights=weights)
    return base_model




def generate_modefied_model():
    baseModel = get_base_model_instance('imagenet', IMG_SHAPE)

    model = tf.keras.Sequential(
    [
     baseModel,
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(1024, activation="relu"),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(7, activation="softmax")]
     )
    
    return model



