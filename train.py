import argparse
import sys

sys.path.insert(0, './utils/')

from os.path import join

from settings import vars
from models.vad import VAD

best_weights = vars.BEST_WEIGHT_PATH

model = VAD(vars, 'experiment', (512, 136, 1))
model.train()
