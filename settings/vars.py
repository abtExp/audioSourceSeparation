from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam

from os import listdir, mkdir
from os.path import isdir
import sys
sys.path.insert(0, './utils/')

from sound_utils import *
from data_loaders import *
import numpy as np


AUDIO_OUTPUT_PATH = './data/out/audio/'
VIDEO_OUTPUT_PATH = './data/out/video/'

AUDIO_TRAIN_FILES_PATH_VOICE = './data/sound/mir/vocals/'
AUDIO_TRAIN_FILES_PATH_FULL_SAMPLES = './data/sound/mir/full_samples/'
AUDIO_TRAIN_FILES_PATH_FULL_FULL = './data/sound/mir/full_full/'

AUDIO_TRAIN_FILES_PATH = './data/sound/vad/train/'
AUDIO_VALID_FILES_PATH = './data/sound/vad/valid/'

INPUT_FILE_TYPE = 'wav'

TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
# TEST_EPOCHS = 10

AUDIO_BATCH_SIZE = 16
VALID_AUDIO_BATCH_SIZE = 2
MAX_SEQUENCE_LENGTH = 4
MAX_TIME = 2.16 #2.154  #Pretty Odd Numbers, here, just to make the model possible
SEQUENCE_FEATURES = 21
# FRAME_RATE = 44000 # Next Option is 22k
FRAME_RATE = 16000
SAMPLE_STEP_SIZE = 2
N_FFT = 1023
FRAMES_PER_BUFFER = 1024
N_OVERLAP = FRAMES_PER_BUFFER // 2
N_BINS = FRAMES_PER_BUFFER // 2 + 1
DATA_LOADER = AUDIO_DATA_LOADER
LABEL_TYPE = 'mask'
DEBUG = False
USE_CLOUD = False

PHASE_ITERATIONS = 10
# BEST_WEIGHT_PATH = './checkpoints/checkpoints_conv_12/weights.99-0.49.hdf5'
# BEST_WEIGHT_PATH = './checkpoints/checkpoints_conv_15/weights.74-0.57.hdf5'
# BEST_WEIGHT_PATH = './checkpoints/checkpoints_conv_18/weights.13-0.23.hdf5'
BEST_WEIGHT_PATH = './checkpoints/checkpoints_custom_0/weights.51-0.58.hdf5'

MODEL_IMAGE_PATH = './model_images/'

def get_callbacks(model='enet', max_time=MAX_TIME):
	all_checks = listdir('./checkpoints/')
	all_logs = listdir('./logs/')
	counter = 0
	max = -1

	for folder in all_checks:
			if 'checkpoints_{}'.format(model) in folder:
					if int(folder[folder.rindex('_')+1:]) > max:
							max = int(folder[folder.rindex('_')+1:])

	counter = max+1
	check_path = './checkpoints/checkpoints_{}_{}/'.format(model, counter)
	logs_path = './logs/logs_{}_{}/'.format(model, counter)

	if not isdir(check_path) and not isdir(logs_path):
			mkdir(check_path)
			mkdir(logs_path)


	checkpoint = ModelCheckpoint(check_path+'weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
	earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0)
	tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
	reducelr = ReduceLROnPlateau(monitor='loss', factor=0.02, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

	return [checkpoint, tensorboard, reducelr]