#! /usr/bin/env python

import os
import tensorflow as tf
from model import CharactersModel, NISTDataset, CharacterSetModel, prepare_example, get_argmax
import numpy as np

paths = dict()
for i in range(0x30, 0x32 + 1):
	paths[chr(i)] = '~/datasets/NIST/by_class/%x/hsf_0' % (i)

count = 5000

model = CharacterSetModel(paths, iters = 10, samples = count)

print('PREDICTING...')

def predict(val):
	x = []
	x.append(prepare_example(os.path.expanduser('~/datasets/NIST/by_class/%s/hsf_0/hsf_0_02194.png' % (val))))
	prediction = get_argmax(model.predict(np.array(x)))
	print(prediction)

predict('30')
predict('31')
predict('32')