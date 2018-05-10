
# coding: utf-8

# In[368]:


import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
from keras.models import load_model
from keras import backend as K
from keras.layers import GRU
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from tensorflow.python.platform import gfile


# global variables
NB_EPOCH = 50
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
AUDIO_ROWS, AUDIO_COLS = 99, 161
NB_CLASSES = 10
DROPOUT = 0.3
INPUT_SHAPE = (AUDIO_ROWS, AUDIO_COLS)
label_to_index_map = {}




# define a LeNet network
class LeNet:
    def build(input_shape, classes):
        model = Sequential()
        
        model.add(Conv1D(20, kernel_size=5, padding='same', input_shape=input_shape)) #layer 1
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
#         model.add(Dropout(DROPOUT))
        
        model.add(Conv1D(50, kernel_size=5, border_mode='same')) #layer 2
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(DROPOUT))
        
        model.add(Conv1D(100, kernel_size=5, border_mode='same')) #layer 3
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
#         model.add(Dropout(DROPOUT))
        
        model.add(Conv1D(200, kernel_size=5, border_mode='same')) #layer 4
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(DROPOUT))

        model.add(GRU(256, return_sequences=True))
        model.add(Dropout(DROPOUT))

        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(DROPOUT))
        
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(DROPOUT))

        model.add(Flatten())
        model.add(Dense(700))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model


def get_label_to_index_map(path):
    labels = os.listdir(path)
    index = 0
    for l in labels:
        label_to_index_map[l] = index
        index += 1

# one hot encoding 
def get_one_hot_encoding(label):
    encoding = [0] * NB_CLASSES
    encoding[label_to_index_map[label]] = 1
    return encoding



def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1000))
    noverlap = int(round(step_size * sample_rate / 1000))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# Wave and spectrogram of a wav file
def get_spectrogram(path):
    sample_rate, samples = wavfile.read(path)
    freqs, times, spectrogram = log_specgram(samples,sample_rate)
    spectrogram.resize(99, 161)
    return spectrogram


# Mel power spectrogram of a wav file
def get_melpower_spectrogram(path):
    sample_rate, samples = wavfile.read(path)
    samples = samples.astype(float)
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    if(log_S.shape[1]!=32):
        new_array = np.zeros((128, 32))
        new_array[:,:log_S.shape[1]] = log_S
        print(new_array.shape)
        print("==========================================")
        return new_array
    else:
        print(log_S.shape)
        print("----------------------------------------")
        return log_S



#loads train and test data 
def load_data(curr_path):
    #load data file by file in trainX and corresponding label in OHE format in trainY
    train_path = curr_path+"/train"
    train_path = os.path.join(train_path, "*", '*.wav')
    train_waves = gfile.Glob(train_path)
    train_X = []
    train_Y = []
    for w in train_waves:
        _, label = os.path.split(os.path.dirname(w))
        print(w, "-----", label)
        train_X.append(get_spectrogram(w))
        train_Y.append(get_one_hot_encoding(label))
        print("TRAIN : ",w," -- done!")
    
    #do similar as above for test folder
    test_X = [] 
    test_Y = []
    test_path = curr_path+"/test"
    test_path = os.path.join(test_path, "*", '*.wav')
    test_waves = gfile.Glob(test_path)
    for w in test_waves:
        _, label = os.path.split(os.path.dirname(w))
        print(w, "-----", label)
        test_X.append(get_spectrogram(w))
        test_Y.append(get_one_hot_encoding(label))
        print("TEST : ",w," -- done!")
    
    return train_X, train_Y, test_X, test_Y


curr_path = os.getcwd()
get_label_to_index_map(curr_path+"/train")

train_X, train_Y, test_X, test_Y = load_data(curr_path)
print(len(train_X))
print(len(train_Y))
print(len(test_X))
print(len(test_Y))

train_X = np.asarray(train_X)
train_Y = np.asarray(train_Y)
test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
model.save('my_model.h5')

model1 = load_model('my_model.h5')

score = model1.evaluate(test_X, test_Y, verbose=VERBOSE)

print("Test score : ", score[0])
print("Test accuracy : ", score[1])

