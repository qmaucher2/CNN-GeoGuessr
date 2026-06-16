
from numpy import arcsin
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory

from sklearn.model_selection import train_test_split
import os
import kagglehub

print("Imports successful. Beginning scripts...")

###############################################################################
# Download latest version

path = kagglehub.dataset_download("sylshaw/streetview-by-country")

print("Path to dataset files:", path)

dataset_dir = os.path.join(path, "streetview_images")

# Preprocess data


########################
print("Preprocessing successful. Compiling model...")

# Define the model

model = keras.Sequential([
    tf.keras.layers.RandomZoom(height_factor=(0.3, 0.3), seed=19),
    tf.keras.layers.RandomContrast(factor=(0.2, 0.2), seed=19),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[275, 275, 3]),
    layers.MaxPool2D(),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

# +---------------------------------------+
# | Error Calculation for Regressor model |
# +---------------------------------------+

def haver(x):
    return (math.sin(x / 2)) ** 2

def hav_theta(p1, p2):
    return haver(p2[1] - p1[1]) + math.cos(p1[1]) * math.cos(p2[1]) * haver(p2[0] - p1[0])
theta = 2 * arcsin(math.sqrt(hav_theta(p1, p2)))

error = 6732.1 * theta  # 6732.1 represents the average radius of the earth

# Compile the model


##################3#

print("Model Compilation successful. Training model...")



# Train the model
batch_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="../Checkpoints/best_model05.keras",
    save_weights_only=False,
    save_freq=25
)



EPOCHS = 50
history = model.fit(
    ds_train_cl,
    validation_data=ds_valid_cl,
    epochs=EPOCHS,
    initial_epoch=0,
    callbacks=[batch_checkpoint]
)



# Loss function
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.save()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot();
plt.save()