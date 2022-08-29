import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, Concatenate
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.efficientnet import EfficientNetB0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from IPython.display import Image

import os
import random
from copy import deepcopy
import pickle
import time

np.random.seed(5)


class Finetuing_Pretrained_Model():
    def __init__(self, G_A_bool_arr, input_shape, freezing_layer_flag=0, type='vgg'):

        self.input_shape = input_shape
        self.type_name = type

        if type == 'vgg':
            self.base_model = VGG16(input_shape=self.input_shape+(3,), include_top=False, weights='imagenet')
        elif type == 'resnet':            
            self.base_model = ResNet50(input_shape=self.input_shape+(3,), include_top=False, weights='imagenet')
        elif type == 'mobilenet':            
            self.base_model = MobileNet(input_shape=self.input_shape+(3,), include_top=False, weights='imagenet')
        else:
            self.base_model = EfficientNetB0(input_shape=self.input_shape+(3,), include_top=False, weights='imagenet')
        
        sample_arr = [True, False]
        self.bool_arr = np.random.choice(sample_arr, size=len(self.base_model.layers))

        if G_A_bool_arr is None:
            self.bool_arr[:freezing_layer_flag] = False
        else:
            self.update_trainable(G_A_bool_arr)
    
    def update_trainable(self, G_A_bool_arr=None):
        if G_A_bool_arr is not None: 
            self.bool_arr = G_A_bool_arr
        self.base_model.trainable = True
        for idx, i in enumerate(self.base_model.layers):
            i.trainable = self.bool_arr[idx]
        
    def forward(self, num_of_class, learning_rate=0.001, one_hot_encoding=True):
        inputs = Input(self.input_shape + (1,))
        first_conv_layer = Conv2D(3, 1, padding='same', activation=None)(inputs)

        x = self.base_model(first_conv_layer, training = False)
        x = Flatten()(x)
        outputs = Dense(num_of_class, activation = 'softmax')(x)

        model = tf.keras.Model(inputs, outputs, name=self.type_name+"_pretrained_model")

        # 'categorical_crossentropy'은 y[0]=[0, 0, 0, 0, 0, 0, 0, 0, 1], y[1, 0, 0, 0, 0, 0, 0, 0, 0]과 같이 one-hot-encoding label일 경우에 사용
        if one_hot_encoding == False:
            model.compile(loss="sparse_categorical_crossentropy", 
            optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), 
            metrics=['accuracy'])
        else:
            model.compile(loss="categorical_crossentropy", 
            optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), 
            metrics=['accuracy'])
        
        model.summary()

        return model
    
    def train_model(self, model, x_train, y_train, x_test, y_test, epochs=20, batch_size=256, checkpoint_path='model_checkpoints_best/checkpoint'):
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint_best_path = checkpoint_path
        checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path,
                                        save_weights_only=True,
                                        save_freq='epoch',
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        verbose=1)
        history = model.fit(x_train, y_train,
                        validation_data = (x_test, y_test),
                        epochs = epochs,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks=[early, checkpoint_best])
        return history

    def load_model(self, model, checkpoint_path='model_checkpoints_best/checkpoint'):
        model.load_weights(checkpoint_path)
        return model

    def eval_model(self, model, x_test, y_test, verbose=2):
        loss, acc = model.evaluate(x_test, y_test, verbose)
        print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
        return loss, acc


def load_picklefile(filename):
    with open(filename, 'rb') as f:
        pair_data = pickle.load(f)

    acc, bool_arr = zip(*pair_data)
    max_point = np.argmax(acc)
    # bool_sum = np.sum(bool_arr[max_point])
    return acc[max_point], bool_arr[max_point]


def load_best_picklefile(folderfath):
    for idx, i in enumerate(os.listdir(folderfath)):
        _, fileExtension = os.path.splitext(i)
        if fileExtension == '.pickle':
            acc, GA_arr = load_picklefile(os.path.join(folderfath, i))

            if idx == 0:
                acc_arr = acc
                bool_arr = GA_arr
            else:
                acc_arr = np.vstack((acc_arr, acc))
                bool_arr = np.vstack((bool_arr, GA_arr))
        else:
            continue

    max_acc = np.argmax(acc_arr)
    print(acc_arr.shape, bool_arr.shape)

    return acc_arr[max_acc], bool_arr[max_acc]


def accuracy_result(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Finetuning preTrained model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def loss_result(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Finetuning preTrained model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()