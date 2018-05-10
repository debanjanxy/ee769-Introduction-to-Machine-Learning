#!usr/bin/python3

import librosa
import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment



curr_path = os.getcwd()
train_path = curr_path+"/train"
filename = "/dog/0a7c2a8d_nohash_0.wav"
filepath = train_path+filename
sample_rate, samples = wavfile.read(filepath)
#sample_rate, samples = wavfile.read("chunk0.wav")
print(sample_rate)
y,sr = librosa.load(filepath)
s = np.abs(librosa.stft(y))
power_to_db = librosa.power_to_db(s**2)
print(np.min(power_to_db))



