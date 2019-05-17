'''
	Classifies between voice activity and other noises

'''
import sys

sys.path.insert(0, './utils/')
sys.path.insert(0, './models/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import os
from os import listdir, path
from os.path import join

import keras
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Reshape, Conv2D, BatchNormalization,\
	UpSampling2D, Concatenate, concatenate, SpatialDropout2D, LeakyReLU, Layer, \
	Activation, ZeroPadding2D, Conv2DTranspose, multiply, Bidirectional, LSTM

from keras.utils import plot_model
from keras.initializers import Ones
from keras.optimizers import Adam, Adadelta
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU
from keras import regularizers
import keras.backend as K

import tensorflow as tf

from pydub import AudioSegment
import librosa

# from utils import *
from sound_utils import *

from model import BASE

class VAD(BASE):
	def __init__(self, vars, model='default', inp_shape=(None, None, 1)):
		self.inp_shape = inp_shape
		self.model_name=model

		super(VAD, self).__init__(vars)

	def compose_model(self):
		inp = Input(shape=self.inp_shape)
		convA = Conv2D(32, 3, activation='relu', padding='same')(inp)
		conv = Conv2D(32, 4, strides=2, activation='relu', padding='same', use_bias=False)(convA)
		conv = BatchNormalization()(conv)

		convB = Conv2D(64, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convB)
		conv = BatchNormalization()(conv)

		conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(128, 3, activation='relu', padding='same', use_bias=False)(conv)
		conv = Conv2D(256, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(256, 3, activation='relu', padding='same', use_bias=False)(conv)
		conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(128, 3, activation='relu', padding='same', use_bias=False)(conv)
		conv = SpatialDropout2D(0.2)(conv)
		conv = BatchNormalization()(conv)
		conv = UpSampling2D((2, 2))(conv)

		conv = Concatenate()([conv, convB])
		conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(64, 3, activation='relu', padding='same', use_bias=False)(conv)
		conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(64, 3, activation='relu', padding='same', use_bias=False)(conv)
		conv = SpatialDropout2D(0.2)(conv)
		conv = BatchNormalization()(conv)
		conv = UpSampling2D((2, 2))(conv)

		conv = Concatenate()([conv, convA])
		conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(8, 3, activation='relu', padding='same')(conv)
		conv = Conv2D(1, 3, activation='sigmoid', padding='same')(conv)

		model = Model(inputs=inp, outputs=conv)
		model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

		return model

	def smooth(self, audio):
		b, a = signal.butter(5, 0.45)
		condn = signal.lfilter_zi(b, a)
		smoothed, _ = signal.lfilter(b, a, audio, zi=condn*audio[0])
		return smoothed

	def process(self, path, mute_mode):
		audio, rate = librosa.load(path, sr=self.vars.FRAME_RATE, mono=True)
		seg = AudioSegment(audio.tobytes(), sample_width=audio.dtype.itemsize, channels=1, frame_rate=self.vars.FRAME_RATE)
		sample_len = len(seg[:self.vars.MAX_TIME*1000].get_array_of_samples())

		total_samples = len(audio)//sample_len
		preds = []

		if mute_mode == 'none':
			return

		for i in range(total_samples):
			sample = audio[i*sample_len:(i+1)*sample_len]
			if len(sample) < sample_len:
				sample = np.concatenate((sample, sample_len-len(sample)), axis=-1)

			sample = np.abs(librosa.stft(sample, n_fft=self.vars.N_FFT))
			sample = np.expand_dims(sample, axis=0)
			sample = np.expand_dims(sample, axis=-1)
			pred = np.squeeze(self.predict(sample))
			if mute_mode == 'voice':
				pred = np.array(pred < 0.3, dtype=np.float32)
			elif mute_mode == 'noise':
				pred = np.array(pred >= 0.7, dtype=np.float32)

			filtered = np.multiply(np.squeeze(sample), pred)
			inp = np.random.randn(sample_len)

			for i in range(self.vars.PHASE_ITERATIONS):
				angle = librosa.stft(inp, n_fft=self.vars.N_FFT)
				if np.shape(angle)[1] < filtered.shape[1]:
					angle = np.concatenate((angle, np.zeros((filtered.shape[0], filtered.shape[1]-np.shape(angle)[1]))), axis=-1)
				full = filtered * np.exp(1j * np.angle(angle))
				inp = librosa.istft(full)

			out = inp
			preds = np.concatenate((preds, out), axis=-1)

		preds = self.smooth(preds)

		fp = path[:path.rindex('/')]
		fl = path[path.rindex('/')+1:path.rindex('.')]

		preds = np.array(preds, dtype=np.float32)

		librosa.output.write_wav('{}/{}_muted_{}.wav'.format(fp, fl, mute_mode), preds, self.vars.FRAME_RATE)
		print('Saved : {}'.format('{}/{}_muted_{}.wav'.format(fp, fl, mute_mode)))