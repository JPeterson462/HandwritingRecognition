#! /usr/bin/env python

import os
import tensorflow as tf
from model import CharactersModel, NISTDataset, CharacterSetModel, prepare_example, get_argmax
import numpy as np

charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
charset = "0123456789"

paths = dict()

for i, v in enumerate(charset):
	paths[v] = '~/datasets/NIST/by_class/%x/hsf_0' % (ord(v))

count = 5000
#count = 500

model = CharacterSetModel(paths, iters = 10, samples = count)

print('EVALUATING...')

model.evaluate()

print('PREDICTING...')

def predict(val):
	x = []
	x.append(prepare_example(os.path.expanduser('~/datasets/NIST/by_class/%x/hsf_0/hsf_0_00194.png' % (val))))
	prediction = get_argmax(model.predict(np.array(x)))
	print(prediction)

for i, v in enumerate(charset):
	predict(ord(v))