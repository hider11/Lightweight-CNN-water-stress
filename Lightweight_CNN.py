import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, Model, optimizers

############load dataset########################################################

data_dir = "./plant_images"

BATCH_SIZE = 4
IMG_SIZE = (224,224)

train_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)


val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

################################################################################

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMG_SHAPE = IMG_SIZE + (3,)

aug_rescale = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.Rescaling(1./255),
  ]
)

num_classes = 3


inputs = keras.Input(shape=IMG_SHAPE)
x = aug_rescale(inputs)
x = layers.Conv2D(32, 3, padding='same', activation=None)(x)


a = layers.AveragePooling2D()(x)
a = layers.ReLU()(a)

m = layers.MaxPooling2D()(x)
m = layers.ReLU()(m)

x = layers.Concatenate()([a,m])

x = layers.Conv2D(48, 3, padding='same', activation=None)(x)

a = layers.AveragePooling2D()(x)
a = layers.ReLU()(a)

m = layers.MaxPooling2D()(x)
m = layers.ReLU()(m)

x = layers.Concatenate()([a,m])

x = layers.Conv2D(64, 3, padding='same', activation=None)(x)

a = layers.AveragePooling2D()(x)
a = layers.ReLU()(a)

m = layers.MaxPooling2D()(x)
m = layers.ReLU()(m)

x = layers.Concatenate()([a,m])

x = layers.Conv2D(64, 3, padding='same', activation=None)(x)

a = layers.AveragePooling2D()(x)
a = layers.ReLU()(a)

m = layers.MaxPooling2D()(x)
m = layers.ReLU()(m)

x = layers.Concatenate()([a,m])

x = layers.Conv2D(104, 5, padding='same', activation=None)(x)
x = layers.ReLU()(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs, out)

model.summary()

################################################################################
adm = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3/200, amsgrad=False)
sgd = optimizers.SGD(learning_rate=1e-4, momentum=0.0, nesterov=False)

epochs = 100

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
]

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=adm,
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    callbacks=my_callbacks,
    epochs=epochs
)

################################################################################
import seaborn as sns

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(20, 20))
sns.set_style('darkgrid')

plt.subplot(2, 1, 1)
plt.tight_layout()
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.grid(True, which='minor', color='white', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')

plt.subplot(2, 1, 2)
plt.tight_layout()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.grid(True, which='minor', color='white', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()
