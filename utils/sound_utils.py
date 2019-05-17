import os
import sys
import cv2
import pydub
import keras
import librosa
import subprocess
import numpy as np
from os import listdir
from os.path import join
from pydub import AudioSegment
import matplotlib.pyplot as plt
from librosa.core import load, resample
from sidekit.frontend.io import read_wav
from sidekit.frontend.features import mfcc


# Load single channel 16k Hz wav files
def preprocess(file, load_type='fp'):
	if load_type == 'fp':
		sig, framerate, sample_width = read_wav(file)
	else:
		sig, framerate, sample_width = file
	sig *= (2**(15-sample_width))
	_, loge, _, mspec = mfcc(sig.astype(np.float32), get_mspec=True)
	return [sig, framerate, sample_width], mspec, loge

def convert(wavfile):
	sw = 2
	# try:
	print(wavfile)
	sig, fr, sw = read_wav(wavfile)
	# except:
	# 	try:
			# sig, fr = load(wavfile)
		# except:
		# 	print('Unsupported Format! Must Be in .wav format')

	sig = np.array(sig, dtype=np.float32)
	shape = sig.shape

	if not len(shape) == 1 or (len(shape)==2 and shape[0] == 1):
		# Convert to single channels
		new_sig = []

	if fr != 16000:
		# convert to 16000Hz
		sig = resample(sig, fr, 16000)

	return preprocess([sig, fr, sw], load_type='file')

# Patches the recorded audio over the original muted audio
def combine_audios(muted_path, recorded_path, output_path):
	command = 'ffmpeg -y -i {} -i {} -filter_complex amix=inputs=2:duration=first:dropout_transition=3 {} -nostats -hide_banner -loglevel quiet'.format(muted_path, recorded_path, output_path)
	subprocess.call(command, shell=True)

# For saving the video file without the audio for ease of integration later
def save_only_video(video_path):
	command = "ffmpeg -y -i {} -codec copy -an {}.muted.mp4 -nostats -hide_banner -loglevel quiet".format(video_path, video_path[:video_path.rindex('.')])
	subprocess.call(command, shell=True)

# Extract the audio from the video
def extract_audio(video_path, output_path, frame_rate=160, multiple=False,  audio_format='wav', channels=1):
	if multiple:
		for file in listdir(video_path):
			command = "ffmpeg -y -i {} -ab {}k -ac {} -ar {} -vn {} -nostats -hide_banner -loglevel quiet".format(join(video_path, file), frame_rate, channels, frame_rate*100, join(output_path, '{}.{}'.format(file[:file.index('.')], audio_format)))
			subprocess.call(command, shell=True)
			save_only_video(join(video_path, file))
	else:
		if '.' in output_path and output_path.index('.') != len(output_path):
			out_path = output_path
		else:
			out_path = join(output_path, '{}.{}'.format(video_path[video_path.rindex('/')+1:video_path.rindex('.')], audio_format))

		command = "ffmpeg -y -i {} -ab {}k -ac 1 -ar {} -vn {} -nostats -hide_banner -loglevel quiet".format(video_path, frame_rate, frame_rate*100, out_path)
		subprocess.call(command, shell=True)
		save_only_video(video_path)

# Write Audio to the video
def overwrite_audio(input_path, audio_path, output_path):
	command = "ffmpeg -y -i {} -i {} -shortest -c:v copy -c:a aac -b:a 160k {} -nostats -hide_banner -loglevel quiet".format(input_path, audio_path, output_path)
	subprocess.call(command, shell=True)

def trim_video(vars, path):
	out_path='D:/recast/data/video/trimmed/'
	files = [file for file in listdir(path) if file.endswith('mp4')]
	for file in files:
		command = 'ffmpeg -y -i {} -ss 00:00:00 -t 00:00:{} -async 1 {} -nostats -hide_banner -loglevel quiet'.format(path+file, vars.VIDEO_MAX_LENGTH_IN_SECONDS, out_path+file)
		subprocess.call(command, shell=True)

def _int64_feature(val):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def export_train_files():
	data_path='D:/recast/data/sound/mir/'

	feature_path = data_path+'full_samples/'

	all_feature_files = listdir(feature_path)

	for file in all_feature_files:
		export_file(feature_path+file, 'train/features')

	for file in all_feature_files:
		export_file(feature_path+file, 'train/labels')

def export_test_file(file_dir):
	data_path = 'D:/recast/data/audio/'
	path = data_path+file_dir

	for file in listdir(path):
		export_file(path+file, 'test/file_dir')


def export_file(file, type_='test'):
	export_path='D:/recast/data/export/{}/{}.tfrecords'.format(type_, file[file.rindex('/'):file.rindex('.')])

	audio, rate = librosa.load(file, sr=16000, mono=False)
	audio = np.transpose(audio)

	if 'labels' in type_:
		print('Labels')
		audio = audio[:, 1]
	else:
		audio = (np.sum(audio, axis=1)/2)

	audio = np.array(audio, dtype=np.float32)

	with tf.python_io.TFRecordWriter(export_path) as writer:
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'audio_len':_int64_feature(len(audio)),
					'audio_raw': _bytes_feature(audio.tostring())
				}
			)
		)

		writer.write(example.SerializeToString())
