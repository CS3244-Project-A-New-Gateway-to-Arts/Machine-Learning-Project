import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import tensorflow.keras as tk
from keras import optimizers
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Flatten, Activation, \
    BatchNormalization
import os
# print(os.listdir("../data/resized/"))
from keras_applications.resnet50 import preprocess_input

image_path = '../data/resized/'

def preprocess_image(image_path):
    from keras.applications import vgg19
    img = load_img(image_path)
    new_width = 100
    new_height = 100
    img = img.resize((new_width, new_height))
    img = img_to_array(img)
    img = img.astype(int)
    return img

def select_image(image_path, artists_top):
    selected_paints = []
    for name in artists_top['name'].values:
        for paints in os.listdir(image_path):
            if name in paints:
                selected_paints.append(preprocess_image(image_path+paints))
    print("lengths is ", len(selected_paints))
    return selected_paints

def select_name(artists_top):
    selected_names = artists_top['name'].str.replace(' ', '_').values.tolist()
    return selected_names

def read_data (csv_path):
    artists = pd.read_csv(csv_path)
    return artists

def select_artists(artists):
    artists = artists.sort_values(by=['paintings'], ascending=False)
    artists_top = artists[artists['paintings'] >= 400].reset_index()
    artists_top = artists_top[['name', 'paintings']]

    # artists_top['class_weight'] = max(artists_top.paintings)/artists_top.paintings
    print(artists_top)
    return artists_top

artists = read_data('../data/artists.csv')
artists_top = select_artists(artists)
# selected_paints = select_image(image_path, artists_top)
selected_paints = select_image(image_path, artists_top)
selected_names = select_name(artists_top)

batch_size = 16
RESNET50_POOLING_AVERAGE = 'avg'
train_input_shape = (224, 224, 3)
NUM_EPOCH = 50
NUM_CLASSES = artists_top.shape[0]
print("number of class is ", NUM_CLASSES)
print(selected_names)

data_generator = ImageDataGenerator(validation_split=0.2)

path = '../data/'

train_generator = data_generator.flow_from_directory(path, target_size=train_input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset= 'training',
        shuffle=True,
        classes = selected_names)

validation_generator = data_generator.flow_from_directory(path, target_size=train_input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'validation',
        shuffle=True,
        classes= selected_names)



STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model = Sequential()

model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet', input_shape = train_input_shape))

# for layer in model.layers:
#     layer.trainable = True
#
# X = model.output
# X = Flatten()(X)

model.add(Dense(NUM_CLASSES, activation = 'softmax'))

model.layers[0].rainable = False

print(model.summary())

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

EARLY_STOP_PATIENCE = 3

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')

fit_history = model.fit_generator(
        generator= train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs = NUM_EPOCH,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        shuffle=True,
        callbacks=[reduce_lr]
)

print(fit_history.history.keys())

plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.show()

test_generator = data_generator.flow_from_directory(
    directory = '../data/test/',
    target_size = train_input_shape[0:2],
    batch_size = batch_size,
    class_mode = None,
    shuffle = False,
    seed = 123
)

test_generator.reset()

pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
predicted_class_indices = np.argmax(pred, axis = 1)

TEST_DIR = '../data/test/'
f, ax = plt.subplots(5, 5, figsize = (15, 15))

import cv2

for i in range(0, 25):
    imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    # a if condition else b
    if predicted_class_indices[i] == 1:
        predicted_class = "Vincent"
    elif predicted_class_indices[i] == 2:
        predicted_class = "Edgar"
    else:
        predicted_class = "Pablo"

    ax[i // 5, i % 5].imshow(imgRGB)
    ax[i // 5, i % 5].axis('off')
    ax[i // 5, i % 5].set_title("Predicted:{}".format(predicted_class))

plt.show()