#! /usr/bin/env python

import os
import tensorflow as tf
from model import CharactersModel, NISTDataset, CharacterSetModel, prepare_example, get_argmax
import numpy as np

paths = dict()

minval = 0x30
maxval = 0x39

for i in range(minval, maxval + 1):
	paths[chr(i)] = '~/datasets/NIST/by_class/%x/hsf_0' % (i)

minval = 0x41
maxval = 0x5a

for i in range(minval, maxval + 1):
	paths[chr(i)] = '~/datasets/NIST/by_class/%x/hsf_0' % (i)

minval = 0x61
maxval = 0x7a

for i in range(minval, maxval + 1):
	paths[chr(i)] = '~/datasets/NIST/by_class/%x/hsf_0' % (i)

count = 5000
count = 500

model = CharacterSetModel(paths, iters = 10, samples = count)

print('PREDICTING...')

def predict(val):
	x = []
	x.append(prepare_example(os.path.expanduser('~/datasets/NIST/by_class/%x/hsf_0/hsf_0_02194.png' % (val))))
	prediction = get_argmax(model.predict(np.array(x)))
	print(prediction)

minval = 0x30
maxval = 0x39

for i in range(minval, maxval + 1):
	predict(i)

minval = 0x41
maxval = 0x5a

for i in range(minval, maxval + 1):
	predict(i)

minval = 0x61
maxval = 0x7a

for i in range(minval, maxval + 1):
	predict(i)