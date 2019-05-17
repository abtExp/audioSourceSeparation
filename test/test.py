import argparse
import sys
from os import listdir
sys.path.insert(0, './settings/')
sys.path.insert(0, './models/')


import vars
from vad import VAD

import librosa
import numpy as np

def process(fl):
	model = VAD(vars)
	model.load_weights()

	if '.' not in fl:
		files = [fl+i for i in listdir(fl)]
	else:
		files = [fl]

	for fl in files:
		model.process(fl,'voice')
		model.process(fl,'noise')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file_name', help='path to the file')

	args = parser.parse_args()

	process(args.file_name)

