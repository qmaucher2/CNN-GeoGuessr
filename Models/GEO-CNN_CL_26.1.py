# Imported libraries
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

# Download latest version
path = kagglehub.dataset_download("sylshaw/streetview-by-country")

print("Path to dataset files:", path)

dataset_dir = os.path.join(path, "streetview_images")

# Preprocess data
ds_train_cl = image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=[224, 224],
    interpolation='nearest',
    batch_size=32,
    shuffle=True,
    seed=161,
    subset="training",
    validation_split=0.3
)

ds_valid_cl = image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=[224, 224],
    interpolation='nearest',
    batch_size=32,
    shuffle=False,
    seed=161,
    subset="validation",
    validation_split=0.3
)


def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE

num_classes = len(ds_train_cl.class_names)

ds_train_cl = (
    ds_train_cl
    .map(convert_to_float)
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid_cl = (
    ds_valid_cl
    .map(convert_to_float)
    .prefetch(buffer_size=AUTOTUNE)
)

print("Preprocessing successful. Compiling model...")

# Define the modell

model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[224, 224, 3]),
    layers.MaxPool2D(),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

print("Model Compilation successful. Training model...")

# Train the model
batch_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="C:/Users/quent/OneDrive/GeoML_QM/best_model.keras",
    save_weights_only=False,
    save_freq=25
)

checkpoint_filepath = "C:/Users/quent/OneDrive/GeoML_QM/best_model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="C:/Users/quent/OneDrive/GeoML_QM/best_model.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

EPOCHS = 10
history = model.fit(
    ds_train_cl,
    validation_data=ds_valid_cl,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback,
               batch_checkpoint]
)

# Loss function
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.savefig("loss_graph.png")
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.savefig("accuracy_graph.png")