import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, Concatenate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from IPython.display import Image

import os

np.random.seed(0)

fashion_mnist = keras.datasets.fashion_mnist
((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class Random_Finetune_ResNet50():
    def __init__(self, input_shape):
        self.fitness = 0
        
        IMG_SHAPE = input_shape + (3,)
        self.base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        sample_arr = [True, False]
        bool_arr = np.random.choice(sample_arr, size=len(self.base_model.layers))
        self.base_model.trainable = True
        for idx, i in enumerate(self.base_model.layers):
            i.trainable = bool_arr[idx]
        
    def forward(self, learning_rate=0.001):
        inputs = Input((28, 28, 1))
        resized_x = tf.keras.layers.experimental.preprocessing.Resizing(32, 32)(inputs)
        first_conv_layer = Conv2D(3, 1, padding='same', activation=None)(resized_x)

        x = self.base_model(first_conv_layer, training = False)
        x = Flatten()(x)
        outputs = Dense(10, activation = 'softmax')(x)

        model = tf.keras.Model(inputs, outputs, name="fashion_mnist_resnet50_model")

        # 'categorical_crossentropy'은 y[0]=[0, 0, 0, 0, 0, 0, 0, 0, 1], y[1, 0, 0, 0, 0, 0, 0, 0, 0]과 같이 one-hot-encoding label일 경우에 사용
        model.compile(loss="categorical_crossentropy", 
        optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), 
        metrics=['accuracy'])
        
        return model

def train_model(model, train_data, train_targets, validation_data=(x_test, y_test), epochs=20, batch_size=256):
    
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_data, train_targets,
                      validation_data = validation_data,
                      epochs = epochs,
                      batch_size = batch_size,
                      verbose = 1,
                      callbacks=[early])
    return history

i = 1
n_gen = 1
gene = Random_Finetune_ResNet50((32,32))
model = gene.forward()
history = train_model(model, x_train, y_train, (x_test, y_test), 20, 256)
fitness = history.history['val_accuracy']
score = history.history['val_loss']
gene.fitness = fitness
print('Generation #%s, Genome #%s, Fitness: %s, Score: %s' % (n_gen, i, fitness, score))
