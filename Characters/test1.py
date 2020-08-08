#! /usr/bin/env python

import os
import tensorflow as tf
from model import CharactersModel, NISTDataset, CharacterSetModel, prepare_example
import numpy as np

paths = dict()
for i in range(0x30, 0x32 + 1):
	paths[chr(i)] = '~/datasets/NIST/by_class/%x/hsf_0' % (i)

#paths['30'] = '~/datasets/NIST/by_class/30/hsf_0'
#paths['31'] = '~/datasets/NIST/by_class/31/hsf_0'
#paths['32'] = '~/datasets/NIST/by_class/32/hsf_0'

count = 5000

#count = 1000

#dataset = NISTDataset(paths, samples = count)
# 600 examples * 62 characters = 37200 examples
# 100 examples * 62 characters = 6200 exampless

#cutoff = int(0.9 * len(paths) * count)

#model = CharactersModel(dataset.x_train[:cutoff], dataset.y_train[:cutoff])
model = CharacterSetModel(paths, iters = 10, samples = count)

#results = model.evaluate(dataset.x_train[cutoff:], dataset.y_train[cutoff:])
#results = model.evaluate(samples = int(count/5))

#print(results)

print('PREDICTING...')

def predict(val):
	x = []
	x.append(prepare_example(os.path.expanduser('~/datasets/NIST/by_class/%s/hsf_0/hsf_0_02194.png' % (val))))
	prediction = model.predict(np.array(x))
	print(prediction)

predict('30')
predict('31')
predict('32')

#print(np.argmax(prediction, axis = -1))