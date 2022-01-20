import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import os
import h5py

path = os.path.join(os.getcwd(), "dataset/input/")

hf = h5py.File(path + "train_snoring_5s.h5", 'r')

x_train = np.array(hf.get('spectrograms'))
y_train = np.array(hf.get('targets')).astype(np.int8)
hf.close()
print(x_train.shape)
print(y_train.shape)

hf = h5py.File(path + "test_snoring_5s.h5", 'r')

x_test = np.array(hf.get('spectrograms'))
y_test = np.array(hf.get('targets')).astype(np.int8)
hf.close()
print(x_test.shape)
print(y_test.shape)

print(x_train.dtype)
print(y_train.dtype)
print(x_test.dtype)
print(y_test.dtype)

print(x_train[0])
print(y_train[0])

## parameter setting
bands = 129
frames = 155
feature_size = 3000
learning_rate = 0.001
training_epochs = 100
batch_size = 128
num_of_class = 2
channel=1

x_train = np.reshape(x_train,(-1,bands,frames,1))
x_test = np.reshape(x_test,(-1,bands,frames,1))
# If subtract pixel mean is enabled

y_train = utils.to_categorical(y_train, num_of_class)
y_test = utils.to_categorical(y_test, num_of_class)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Input, BatchNormalization, ELU
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Create the base model from the pre-trained model Resnet-50
# ResNet50 불러오기에서 include_top=을 True가 아닌 False로 둠으로써 사전학습된 모델의 최상층 분류기를 사용하지 않겠다고 설정
IMG_SHAPE = (129, 155) + (3,)
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# base_model.trainable = False를 통해 사전학습된 resnet이 기존에 기억하던 weights를 손상주지 않기 위해 동결
base_model.trainable = False

# 변형된 전이학습 모델
# intput
inputs = Input(shape=(bands, frames, channel), name='inputs')

# resized_x = tf.keras.layers.experimental.preprocessing.Resizing(32, 32)(inputs)
first_conv_layer = Conv2D(3, 1, padding='same', activation=None)(inputs)

x = base_model(first_conv_layer, training = False)
x = Flatten()(x)
outputs = Dense(num_of_class, activation = 'softmax')(x)

model = tf.keras.Model(inputs, outputs, name="resnet50_based_model")

model.summary()
tf.keras.utils.plot_model(model, "resnet50_based_model_with_shape_info.png", show_shapes=True)

# Model compile
model.compile(loss="categorical_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), 
              metrics=['accuracy'])

# early stopping 설정
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Model fit
hist = model.fit(x_train, y_train,
                 validation_data = (x_test, y_test),
                 epochs = training_epochs,
                 batch_size = batch_size,
                 verbose = 1,
                 callbacks=[early])

# hist의 accuracy plt의 plot을 이용하여 출력하는 코드를 작성하세요.
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.savefig('results_resnet50_based_model.png')

res = model.predict(x_test[0:1])
print(res)

plt.bar(range(2), res[0], color='red')
plt.bar(np.array(range(2)) + 0.35, y_test[0])
plt.savefig('test1.png')

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(loss, acc)


# model save and load
model.save("./model/resnet50_based_model.h5")    # 모델과 weights를 전부 저장

new_model = tf.keras.models.load_model('./model/resnet50_based_model.h5')

res = new_model.predict( x_test[3:4] ) 
print(res.shape)
print(res[0])
plt.bar(range(2), res[0], color='red')
plt.bar(np.array(range(2)) + 0.35, y_test[3])
plt.savefig('test2.png')

loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
print(loss, acc)