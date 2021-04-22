import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def training_data(data_dir, batch_size, img_height, img_width, colour_option):
      train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      #RGB for colour images grayscale for b&w
      color_mode=colour_option,
      validation_split=0.2,
      subset="training",
      shuffle = True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
      return train_ds

def validation_data(data_dir, batch_size, img_height, img_width, colour_option):
      val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      #RGB for colour images grayscale for b&w
      color_mode=colour_option,
      validation_split=0.2,
      subset="validation",
      shuffle = True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
      return val_ds

def model_build(img_height, img_width, colour_chanels, hl1, hl2):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='swish', input_shape=(img_height, img_width, colour_chanels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='swish'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='swish'))
    model.add(layers.Flatten())
    model.add(layers.Dense(hl1, activation='swish'))
    model.add(layers.Dense(hl2, activation='swish'))
    model.add(layers.Dense(num_classes))
    #model.summary()
    model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    return model

strategy = tf.distribute.experimental.MirroredStrategy()
time_start = tf.timestamp()
batch_size = 32
num_classes = 10
num_epochs = 5
hl1 = 60
hl2 = 40
img_height = 28
img_width = 28
colour_chanels = 1
colour_option = 'grayscale'
#title = "Digits "
#data_dir = "/home/chris/tensorflow/datasets/Digits"
#save_dir = "/home/chris/tensorflow/saved/Digits"
title = "Fashion "
data_dir = "/home/chris/tensorflow/datasets/Fashion"
save_dir = "/home/chris/tensorflow/saved/Fashion"
#title = "EuroSat "
#data_dir = "/home/chris/tensorflow/datasets/EuroSat"
#save_dir = "/home/chris/tensorflow/saved/EuroSat"
multi_training = training_data(data_dir, batch_size, img_height, img_width, colour_option)
multi_validation = validation_data(data_dir, batch_size, img_height, img_width, colour_option)
AUTOTUNE = tf.data.AUTOTUNE
multi_training = multi_training.cache().prefetch(buffer_size=AUTOTUNE)
multi_validation = multi_validation.cache().prefetch(buffer_size=AUTOTUNE)
with strategy.scope():
    multi_model = model_build(img_height, img_width, colour_chanels, hl1, hl2)
    history = multi_model.fit(
        multi_training,
        validation_data=multi_validation,
        epochs=num_epochs
        )
time_end = tf.timestamp()
evaluation = str(multi_model.evaluate(multi_validation,verbose = 2))
last_chars = evaluation[ 20: 27: 1]
print(last_chars)
total_time = "Total Time%8.2f Seconds" % float(time_end - time_start)
print(total_time)
hl1_str = str(hl1)
hl2_str = str(hl2)
file_extention = ".png"
epochs_str = str(num_epochs)
filename = title + "Epochs = " + epochs_str + " Hidden layers " + hl1_str + " " + hl2_str + " " + total_time + " " + last_chars + file_extention
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(filename)
multi_model.save(save_dir)
