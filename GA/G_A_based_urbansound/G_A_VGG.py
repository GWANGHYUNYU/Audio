import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, Concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

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

# Load dataset

load_path = 'D:\GH\Audio\dataset\\UrbanSound8K\\audio\\np_data'

'''
loaded = np.load('파일명.npz', allow_pickle=True)
(128, 173, 1)
'''

train_path = os.path.join(load_path, 'fold1train.npy')
test_path = os.path.join(load_path, 'fold1test.npy')

train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)

x_train = train_data[:, -1]
y_train = train_data[:, 1]
x_test = test_data[:, -1]
y_test = test_data[:, 1]

x_train = np.stack(x_train, axis=0)
x_test = np.stack(x_test, axis=0)

x_train = x_train[...,np.newaxis]
x_test = x_test[...,np.newaxis]
y_train = y_train.reshape((-1,))
y_test = y_test.reshape((-1,))

# 배열의 원소 개수만큼 인덱스 배열을 만든 후
# 무작위로 뒤섞어 줍니다.
idx_train = np.arange(x_train.shape[0])
idx_test = np.arange(x_test.shape[0])
# print(idx)
np.random.shuffle(idx_train)
np.random.shuffle(idx_test)

x_train_shuffle = x_train[idx_train]
y_train_shuffle = y_train[idx_train]
x_test_shuffle = x_test[idx_test]
y_test_shuffle = y_test[idx_test]

# TensorFlow에서 읽을 수 있는 np.astype으로 변경
x_train = np.asarray(x_train_shuffle).astype(np.float64)
y_train = np.asarray(y_train_shuffle).astype(np.int64)
x_test = np.asarray(x_test_shuffle).astype(np.float64)
y_test = np.asarray(y_test_shuffle).astype(np.int64)

print('Train/Test Dataset Shape')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


class Random_Finetune_VGG():
    def __init__(self, input_shape, freezing_layer_flag):

        self.fitness = 0
        # self.loss = 1000
        
        IMG_SHAPE = input_shape + (3,)
        self.base_model = VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        # self.base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        sample_arr = [True, False]
        self.bool_arr = np.random.choice(sample_arr, size=len(self.base_model.layers))
        self.bool_arr[:freezing_layer_flag] = False
        self.update_trainable()
        # self.base_model.trainable = True
        # for idx, i in enumerate(self.base_model.layers):
        #     i.trainable = self.bool_arr[idx]
    
    def update_trainable(self, bool_arr=None):
        if bool_arr is not None: 
            self.bool_arr = bool_arr
        self.base_model.trainable = True
        for idx, i in enumerate(self.base_model.layers):
            i.trainable = self.bool_arr[idx]
        
    def forward(self, learning_rate=0.001):
        inputs = Input((128, 173, 1))
        first_conv_layer = Conv2D(3, 1, padding='same', activation=None)(inputs)

        x = self.base_model(first_conv_layer, training = False)
        x = Flatten()(x)
        outputs = Dense(10, activation = 'softmax')(x)

        model = tf.keras.Model(inputs, outputs, name="UrbanSound8K__VGG_model")

        # 'categorical_crossentropy'은 y[0]=[0, 0, 0, 0, 0, 0, 0, 0, 1], y[1, 0, 0, 0, 0, 0, 0, 0, 0]과 같이 one-hot-encoding label일 경우에 사용
        model.compile(loss="sparse_categorical_crossentropy", 
        optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), 
        metrics=['accuracy'])
        
        return model
    
    def train_model(self, model, train_data, train_targets, validation_data=(x_test, y_test), epochs=20, batch_size=256):
    
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        checkpoint_best_path = 'model_checkpoints_best/checkpoint'
        checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path,
                                        save_weights_only=True,
                                        save_freq='epoch',
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        verbose=1)
        history = model.fit(train_data, train_targets,
                        validation_data = validation_data,
                        epochs = epochs,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks=[early])
        return history

def crossover_sequentail(iteration, winner_acc, winner_bool_arr, class_instance_arr, freezing_flag):
    
    # max part는 추후 수정 예정
    max = len(winner_acc)
    if max < iteration:
        for i in range(iteration-max):
            winner_acc = np.append(winner_acc, winner_acc[i])
            winner_bool_arr = np.append(winner_bool_arr, winner_bool_arr[i])

    for i, genome in enumerate(class_instance_arr):

        a_genome = deepcopy(winner_bool_arr[i])
        b_genome = deepcopy(winner_bool_arr[i-1])

        new_genome = []
        cut = np.random.randint(freezing_flag, len(winner_bool_arr[0]))
        new_genome.extend(a_genome[:cut])
        new_genome.extend(b_genome[cut:])

        children = np.asarray(new_genome)
        genome.update_trainable(bool_arr=children)
        avg_fitness = (winner_acc[i] + winner_acc[i-1])/2
        genome.fitness = avg_fitness

        print('Generation #%s, Crossover_sequence Genome #%s Created Bool_Array: %s layers, Predicted Fitness: %s, Done' % (n_gen, i, len(children), genome.fitness))


def crossover_random(iteration, winner_acc, winner_bool_arr, class_instance_arr, freezing_flag):

    # max part는 추후 수정 예정
    max = len(winner_acc)
    if max < iteration:
        for i in range(iteration-max):
            winner_acc = np.append(winner_acc, winner_acc[i])
            winner_bool_arr = np.append(winner_bool_arr, winner_bool_arr[i])
    
    for i, genome in enumerate(class_instance_arr):

        flag = np.random.randint(freezing_flag, len(winner_acc), size=2)

        a_genome = deepcopy(winner_bool_arr[flag[0]])
        b_genome = deepcopy(winner_bool_arr[flag[-1]])

        new_genome = []
        cut = np.random.randint(0, len(winner_bool_arr[0]))
        new_genome.extend(a_genome[:cut])
        new_genome.extend(b_genome[cut:])

        children = np.asarray(new_genome)
        genome.update_trainable(bool_arr=children)
        avg_fitness = (winner_acc[i] + winner_acc[i-1])/2
        genome.fitness = avg_fitness

        print('Generation #%s, Crossover_Random Genome #%s Created Bool_Array: %s layers, Predicted Fitness: %s, Done' % (n_gen, i, len(children), genome.fitness))

def mutation(winner_bool_arr, class_instance_arr, freezing_layer_flag):
    
    for i, genome in enumerate(class_instance_arr):

        print('Generation #%s, Genome #%s, Predicted Fitness: %s' % (n_gen, i, genome.fitness))
        
        mutation_copy_arr = deepcopy(genome.bool_arr)

        if np.random.uniform(0,1) < PROB_MUTATION:

            flag = np.random.randint(0, len(winner_bool_arr[0]))
            flag = round(flag*PROB_MUTATION)

            mutation_arr = random.sample(range(freezing_layer_flag, len(winner_bool_arr[0])), flag)
            mutation_arr_sort = sorted(mutation_arr)
            
            for idx in mutation_arr_sort:
                boolen = mutation_copy_arr[idx]
                if boolen == True:
                    mutation_copy_arr[idx] = False
                else:
                    mutation_copy_arr[idx] = True

            genome.update_trainable(bool_arr=mutation_copy_arr)
            
            print('Generation #%s, Genome #%s, Mutation Happened!!! \t size: %s, Mutation_Array: %s, Done' % (n_gen, i, flag, mutation_arr_sort))

# Parameters
IMG_SHAPE = (128, 173)

N_POPULATION = 10
N_BEST = 5
N_CHILDREN = 5
PROB_MUTATION = 0.04

Freezing_layer = 0      # round(Total_layer/2)

lr = 0.0001
epoch = 5
batch_size = 32
save_path = 'D:\GH\Audio\GA\pickle_data\\UrbanSound8K\\0623_VGG_NO_FREEZE'

# generate 1st population
genomes = [Random_Finetune_VGG(IMG_SHAPE, Freezing_layer) for _ in range(N_POPULATION)]
nw_genomes = [Random_Finetune_VGG(IMG_SHAPE, Freezing_layer) for _ in range(N_POPULATION)]

n_gen = 0

first_accuracy = np.array([])
first_bool_arr = []
for i, genome in enumerate(genomes):
    print("===== Generaton #%s\tGenome #%s : Fitness %s =====" % (n_gen, i, genome.fitness))
    # print(type(genome.bool_arr))
    # print(genome.bool_arr)
    first_accuracy = np.append(first_accuracy, genome.fitness)
    # bool_arr = np.append(bool_arr, genome.bool_arr)
    first_bool_arr.append(genome.bool_arr)

first_bool_arr = np.asarray(first_bool_arr)
data = zip(first_accuracy, first_bool_arr)

# save
filename1 = '0_Generation_Bool_Arr.pickle'
filepath = os.path.join(save_path, filename1)

with open(filepath, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


while True:
    startGenomes = time.time()  # 시작 시간 저장

    n_gen += 1
    print("====================== Generaton #%s\t START ======================" % (n_gen))
    for i, genome in enumerate(genomes):

        genome = genomes[i]
        model = genome.forward(lr)
        history = genome.train_model(model, x_train, y_train, (x_test, y_test), epoch, batch_size)
        fitness = history.history['val_accuracy']
        sorted_fitness = sorted(fitness, reverse=True)
        genome.fitness = sorted_fitness[0]

        print('Generation #%s, Genome #%s, Fitness: %s, Best Fitness: %s' % (n_gen, i, fitness, genome.fitness))

    print("===== Generaton #%s\t Processing time : %s seconds =====" % (n_gen, time.time() - startGenomes))    # 현재시각 - 시작시간 = 실행 시간

    for i, genome in enumerate(genomes):
        print("===== Generaton #%s\t%s th Fitness %s =====" % (n_gen, i, genomes[i].fitness))

    # if best_genomes is not None:
    #     genomes.extend(best_genomes)
    genomes.sort(key=lambda x: x.fitness, reverse=True)

    print('===== Generaton #%s\tBest Fitness %s =====' % (n_gen, genomes[0].fitness))

    # 우성 genomes 를 만드는 과정
    accuracy = np.array([])
    bool_arr = []

    for i, genome in enumerate(genomes):
        print("===== Generaton #%s\tGenome #%s : Fitness %s =====" % (n_gen, i, genome.fitness))
        # print(type(genome.bool_arr))
        # print(genome.bool_arr)
        accuracy = np.append(accuracy, genome.fitness)
        # bool_arr = np.append(bool_arr, genome.bool_arr)
        bool_arr.append(genome.bool_arr)

    bool_arr = np.asarray(bool_arr)

    print(" 우성 genome에 대한 Freezing, Trainable Layer 서치 완료 ")
    # print(accuracy.shape)
    # print(bool_arr.shape)

    # 우성 bool_arr
    winner_acc = accuracy[:N_BEST]
    winner_bool_arr = bool_arr[:N_BEST]

    # CROSSOVER with Sequantial
    crossover_sequentail(N_CHILDREN, winner_acc, winner_bool_arr, nw_genomes[:N_CHILDREN], Freezing_layer)
    crossover_sequentail(N_CHILDREN, winner_acc, winner_bool_arr, nw_genomes[N_CHILDREN:], Freezing_layer)

    # mutation
    mutation(winner_bool_arr, nw_genomes, Freezing_layer)

    # 유전 연산 결과를 업데이트하는 과정
    process_accuracy = np.array([])
    process_bool_arr = []
    for i, genome in enumerate(nw_genomes):
        print("===== Generaton #%s\tGenome #%s : Fitness %s =====" % (n_gen, i, genome.fitness))
        # print(type(genome.bool_arr))
        # print(genome.bool_arr)
        process_accuracy = np.append(process_accuracy, genome.fitness)
        # bool_arr = np.append(bool_arr, genome.bool_arr)
        process_bool_arr.append(genome.bool_arr)

    process_bool_arr = np.asarray(process_bool_arr)

    print("===== Generaton #%s\t Total Processing time : %s seconds =====" % (n_gen, time.time() - startGenomes))    # 현재시각 - 시작시간 = 실행 시간

    print(" 유전연산에 대한 Freezing, Trainable Layer 서치 완료 및 Next Generation 준비 ")
    # print(process_accuracy.shape)
    # print(process_bool_arr.shape)

    for i, genome in enumerate(genomes):
        genome.update_trainable(bool_arr=process_bool_arr[i])
        genome.fitness = process_accuracy[i]

    # genomes.sort(key=lambda x: x.fitness, reverse=True)
    print('Generation #%s, Done' % (n_gen))
    data = zip(process_accuracy, process_bool_arr)

    # save
    filename = str(n_gen) + '_Generation_Bool_Arr.pickle'
    filepath = os.path.join(save_path, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)