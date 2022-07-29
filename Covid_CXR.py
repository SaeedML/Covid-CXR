###################################################### Imports ##############################################

import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

######################################## Reproducability and setting matplotlib defaults ####################
def set_seed(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(1234)

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells

########################################## Load training, validation and testing sets ########################

ds_train_ = image_dataset_from_directory(
    '/Users/saeed/Desktop/COVID_CXR/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    '/Users/saeed/Desktop/COVID_CXR/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)
ds_test_ = image_dataset_from_directory(
    '/Users/saeed/Desktop/COVID_CXR/test',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_test = (
    ds_test_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

############################################## Loading pretrained keras model ##############################

pretrained_base = tf.keras.models.load_model(
    '/Users/saeed/Desktop/model_base',
)
pretrained_base.trainable = False

############################################## Creating CNN model ##########################################

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

model_covid = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(12, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

############################################## Compiling of the model ########################################

model_covid.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy'],
)

############################################### Fitting model on the dataset #################################

history = model_covid.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=5,
    verbose=1,
)
############################################# Visualizion of the model's metrics ###############################

import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()

############################################## Evaluation of model on the Testing set ###########################

model_covid.evaluate(x=ds_test)

