
import os
from os.path import isdir, join
from pathlib import Path
from scipy.io import wavfile

from keras.models import load_model
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
# import seaborn as sns
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
NB_EPOCH = 5
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
AUDIO_ROWS, AUDIO_COLS = 99, 161
NB_CLASSES = 10
DROPOUT = 0.3
INPUT_SHAPE = (AUDIO_ROWS, AUDIO_COLS)
label_to_index_map = {}



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



def get_spectrogram(path):
    sample_rate, samples = wavfile.read(path)
    freqs, times, spectrogram = log_specgram(samples,sample_rate)
    spectrogram.resize(99, 161)
   
    return spectrogram



model = load_model('my_model.h5')

# model.compile(loss='binary_crossentropy',
#             optimizer='rmsprop',
#              metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

test_X = []
curr_path = os.getcwd()
test_path = curr_path+"/predict/"
test_path = os.path.join(test_path, '*.wav')
test_wave = gfile.Glob(test_path)
#print(test_wave)
for w in test_wave:
	print(w)
	test_X.append(get_spectrogram(w))

test_X = np.asarray(test_X)


classes = model.predict_classes(test_X)
#print(len(test_X))

print (classes)
