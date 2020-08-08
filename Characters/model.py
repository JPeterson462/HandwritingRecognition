import tensorflow as tf
import os, random
from PIL import Image
import numpy as np
import operator

def get_argmax(data):
	return max(data.items(), key=operator.itemgetter(1))

def prepare_example(filename):
	img = Image.open(filename)
	img = img.resize((28, 28))
	mat = np.array(img)
	compressed = (mat[:, :, 0] + mat[:, :, 1] + mat[:, :, 2]) / 3.0
	#print(compressed.shape)
	return compressed

class NISTDataset(object):
	def __init__(self, paths, samples = -1):
		self.x_train = []
		self.y_train = []
		for digit, path in paths.items():
			count = 0
			full_path = os.path.expanduser(path)
			for filename in os.listdir(full_path):
				if samples is not None and samples > 0 and count > samples:
					break
				count += 1
				self.x_train.append(prepare_example(full_path + '/' + filename))
				self.y_train.append(digit)
		self.y_train = np.array(self.y_train)
		#print(self.x_train)
		#print(self.y_train)

class CharactersModel(object):
	def __init__(self, x_train, y_train, iters = 5, normalize = True):
		self.model = tf.keras.models.Sequential([
		  tf.keras.layers.Flatten(input_shape=(28, 28)),
		  tf.keras.layers.Dense(128, activation='relu'),
		  tf.keras.layers.Dropout(0.2),
		  tf.keras.layers.Dense(10, activation='softmax')
		])
		self.model.compile(optimizer='adam',
		              loss='sparse_categorical_crossentropy',
		              metrics=['accuracy'])
		if normalize:
			x_train = np.array(x_train) / 255.0
		self.model.fit(x_train, y_train, epochs = iters)

	def evaluate(self, x_test, y_test, normalize = True):
		if normalize:
			x_test = np.array(x_test) / 255.0
		return self.model.evaluate(x_test, y_test, return_dict=True)

	def predict(self, x, normalize = True):
		if normalize:
			x = np.array(x) / 255.0
		return self.model.predict(x)

class CharacterSetModel(object):
	def __init__(self, paths, iters = 5, samples = 1000):
		self.paths = paths
		self.models = dict()
		for value, path in paths.items():
			print('Training model for %s with %d samples' % (value, samples))
			samples_per = int(samples / (len(paths) - 1)) * 3
			path_set = list(paths.values())
			path_set.remove(path)
			x_train = []
			x_train.extend(self.get_n_labeled_examples(path, samples, 1, use_random = False))
			x_train.extend(self.get_n_labeled_examples_from_each(path_set, samples_per, 0, use_random = True))
			y_train = []
			y_train.extend([1] * samples)
			y_train.extend([0] * (len(x_train) - len(y_train)))
			x_train = np.array(x_train)
			y_train = np.array(y_train)
			print(x_train.shape)
			print(y_train.shape)
			self.models[value] = CharactersModel(x_train, y_train, iters, normalize = True)

	def evaluate(self, samples = 200):
		results = []
		for value, path in self.paths.items():
			print('Evaluating model for %s with %d samples' % (value, samples))
			samples_per = int(samples / (len(self.paths) - 1))
			path_set = list(self.paths.values())
			path_set.remove(path)
			x_test = []
			x_test.extend(self.get_n_labeled_examples(path, samples, 1, use_random = True))
			x_test.extend(self.get_n_labeled_examples_from_each(path_set, samples_per, 0, use_random = True))
			y_test = []
			y_test.extend([1] * samples)
			y_test.extend([0] * (len(x_test) - len(y_test)))
			x_test = np.array(x_test)
			y_test = np.array(y_test)
			results.append(self.models[value].evaluate(x_test, y_test, normalize = True))
		return results

	def predict(self, x, normalize = True):
		probabilities = dict()
		for value, model in self.models.items():
			output = model.predict(x, normalize)
			probabilities[value] = output[0][1]
		total = np.sum(list(probabilities.values()))
		for key in probabilities.keys():
			probabilities[key] = probabilities[key] / total
		return probabilities

	def get_n_labeled_examples_from_each(self, paths, n, label, use_random = False):
		x_train = []
		for path in paths:
			x_train.extend(self.get_n_labeled_examples(path, n, label, use_random))
		return x_train
		
	def get_n_labeled_examples(self, path, n, label, use_random = False):
		paths = os.listdir(os.path.expanduser(path))
		n = min(n, len(paths))
		print('Paths: %d, Samples: %d' % (len(paths), n))
		if use_random:
			paths = random.sample(paths, n)
		else:
			paths[:n]
		x_train = [prepare_example(os.path.expanduser(path) + '/' + filename) for filename in paths]
		return x_train