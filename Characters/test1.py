#! /usr/bin/env python

import tensorflow as tf
from model import CharactersModel, NISTDataset

paths = dict()
paths[0] = '~/datasets/NIST/by_class/30/hsf_0'
paths[1] = '~/datasets/NIST/by_class/31/hsf_0'

count = 1500

dataset = NISTDataset(paths, samples = count)
# 600 examples * 62 characters = 37200 examples
# 100 examples * 62 characters = 6200 exampless

cutoff = int(0.9 * len(paths) * count)

model = CharactersModel(dataset.x_train[:cutoff], dataset.y_train[:cutoff])

results = model.evaluate(dataset.x_train[cutoff:], dataset.y_train[cutoff:])

print(results)