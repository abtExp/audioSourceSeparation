import sys
import keras
import librosa
import numpy as np
from os import listdir
import tensorflow as tf
from os.path import join
from scipy.io import wavfile
sys.path.insert(0, './utils/')
from sound_utils import *
from pydub import AudioSegment
from librosa.display import specshow
from sidekit.frontend.features import mfcc


def load_from_tfrecord(file):
	audio_data = []
	example = tf.data.TFRecordDataset(file)
	dict_ = {
		'audio_len':tf.FixedLenFeature([], tf.int64),
		'audio_raw':tf.FixedLenFeature([], tf.string)
	}
	parsed = example.map(lambda x:tf.parse_single_example(x, dict_))
	for audio in parsed:
		audio_data.append(audio['audio_raw'].numpy())

	audio = np.frombuffer(audio_data[0], dtype=np.float32)
	return audio

def data_loader(vars, file_type='wav', mode='train'):
	features_path = vars.AUDIO_TRAIN_FILES_PATH_FULL_SAMPLES
	all_files = listdir(features_path)
	train_x, train_y = [], []

	batch_size = vars.TRAIN_BATCH_SIZE if mode == 'train' else vars.VALID_BATCH_SIZE

	if mode == 'valid':
		all_tx, all_ty = find_files()

	while len(train_x) < batch_size:
		if file_type=='wav':
			if mode == 'valid':
				file_idx = np.random.randint(0, len(all_tx))
				feature = all_tx[file_idx]
				label = all_ty[file_idx]

				start_idx = np.random.randint(0, 120)
				audio, _ = librosa.load(feature, sr=vars.FRAME_RATE, mono=False, offset=start_idx, duration=vars.MAX_TIME)
				lab, _ = librosa.load(label, sr=vars.FRAME_RATE, mono=False, offset=start_idx, duration=vars.MAX_TIME)
				audio = np.transpose(audio)
				lab = np.transpose(lab)
				feature = (np.sum(audio, axis=1, dtype=np.float32)/2)
				target = (np.sum(lab, axis=1, dtype=np.float32)/2)
				feature_sample = feature
				target_sample = target

				sample_len = len(feature_sample)
			else:
				file_idx = np.random.randint(0, len(all_files))
				file = all_files[file_idx]
				audio, _ = librosa.load(join(features_path, file), sr=vars.FRAME_RATE, mono=False)
				audio = np.transpose(audio)
				feature = (np.sum(audio, axis=1)/2)
				target = audio[:, 1]

				feature_seg = AudioSegment(
					feature.tobytes(),
					frame_rate=vars.FRAME_RATE,
					sample_width=feature.dtype.itemsize,
					channels=1
				)

				sample_len = len(feature_seg[:int(vars.MAX_TIME*1000)].get_array_of_samples())

				start_idx = np.random.randint(0, len(audio)-sample_len)

				feature_sample = np.array(feature[start_idx:start_idx+sample_len], dtype=np.float32)
				target_sample = np.array(target[start_idx:start_idx+sample_len], dtype=np.float32)

		else:
			tf.enable_eager_execution()

			all_feature_records = listdir(vars.CLOUD_FEATURE_PATH)
			all_label_records = listdir(vars.CLOUD_LABEL_PATH)

			file_idx = np.random.randint(0, len(all_feature_records))

			feature = all_feature_records[file_idx]
			label = all_label_records[file_idx]

			feature = load_from_tfrecord(vars.CLOUD_FEATURE_PATH+feature)
			target = load_from_tfrecord(vars.CLOUD_LABEL_PATH+label)

			sample_len = 94776

			start_idx = np.random.randint(0, len(feature)-sample_len)

			feature_sample = np.array(feature[start_idx:start_idx+sample_len], dtype=np.float32)
			target_sample = np.array(target[start_idx:start_idx+sample_len], dtype=np.float32)

		feature_sample = np.abs(librosa.stft(feature_sample, n_fft=vars.N_FFT))
		target_sample = np.abs(librosa.stft(target_sample, n_fft=vars.N_FFT))

		if vars.DEBUG:
			specshow(librosa.amplitude_to_db(feature_sample), sr=vars.FRAME_RATE)
			plt.show()

			specshow(librosa.amplitude_to_db(target_sample), sr=vars.FRAME_RATE)
			plt.show()

		mask = np.divide(target_sample, feature_sample+1e-3)
		mask = np.array(mask >= 0.8, dtype=np.float32)
		target_sample = mask

		if vars.DEBUG:
			specshow(librosa.amplitude_to_db(target_sample), sr=vars.FRAME_RATE)
			plt.show()

		stft_shape = [int(1+vars.N_FFT/2), int(1+((sample_len-vars.N_FFT)/(vars.N_FFT//4)))]

		if stft_shape[1]%2 != 0:
			stft_shape = [stft_shape[0], stft_shape[1]-1]

		if np.shape(feature_sample)[1] < stft_shape[1]:
			remaining = stft_shape[1] - np.shape(feature_sample)[1]
			zeros = np.zeros((stft_shape[0], remaining))
			feature_sample = np.concatenate((feature_sample, zeros), axis=-1)
			target_sample = np.concatenate((target_sample, zeros), axis=-1)

			feature_sample = feature_sample[:, :stft_shape[1]]
			target_sample = target_sample[:, :stft_shape[1]]

		train_x.append(feature_sample)
		train_y.append(target_sample)

	train_x = np.expand_dims(train_x, axis=-1)
	train_y = np.expand_dims(train_y, axis=-1)

	return train_x, train_y

def find_files():
	data_path = './DSD100'

	tx_path = data_path+'/Mixtures/Dev/'
	ty_path = data_path+'/Sources/Dev/'

	txx_path = data_path+'/Mixtures/Test/'
	tyy_path = data_path+'/Sources/Test/'

	all_tx_folders = listdir(tx_path)
	all_ty_folders = listdir(ty_path)

	all_txx_folders = listdir(txx_path)
	all_tyy_folders = listdir(tyy_path)

	all_tx = []
	all_ty = []

	avail_xs = []
	avail_ys = []

	avail_xxs = []
	avail_yys = []

	for song in all_tx_folders:
		if len(listdir(tx_path+song)) > 0:
			avail_xs.append(song)

	for vocal in all_ty_folders:
		if 'vocals.wav' in listdir(ty_path+vocal):
			avail_ys.append(vocal)

	for song in all_txx_folders:
		if len(listdir(txx_path+song)) > 0:
			avail_xxs.append(song)

	for vocal in all_tyy_folders:
		if 'vocals.wav' in listdir(tyy_path+vocal):
			avail_yys.append(vocal)

	songs = [song for song in avail_xs if song in avail_ys]
	songgs = [song for song in avail_xxs if song in avail_yys]

	for song in songs:
		all_tx.append(tx_path+song+'/mixture.wav')
		all_ty.append(ty_path+song+'/vocals.wav')

	for song in songgs:
		all_tx.append(txx_path+song+'/mixture.wav')
		all_ty.append(tyy_path+song+'/vocals.wav')

	return all_tx, all_ty


class AUDIO_DATA_LOADER(keras.utils.Sequence):
	def __init__(self, vars, mode):
		self.vars = vars
		self.mode = mode
		self.file_type=vars.INPUT_FILE_TYPE

	def __getitem__(self, index):
		x, y = self.__data_generation([])
		index = np.random.randint(0, len(x))

		x, y = x[index], y[index]

		x = np.expand_dims(x, axis=0)
		y = np.expand_dims(y, axis=0)

		return x, y

	def __len__(self):
		return 100

	def __data_generation(self, l):
		return data_loader(self.vars, self.file_type, self.mode)