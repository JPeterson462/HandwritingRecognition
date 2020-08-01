import tensorflow as tf
import os
from PIL import Image
import numpy as np

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
				self.x_train.append(self.prepare_example(full_path + '/' + filename))
				self.y_train.append(digit)
		self.y_train = np.array(self.y_train)
		#print(self.x_train)
		#print(self.y_train)

	def prepare_example(self, filename):
		img = Image.open(filename)
		img = img.resize((28, 28))
		mat = np.array(img)
		compressed = (mat[:, :, 0] + mat[:, :, 1] + mat[:, :, 2]) / 3.0
		#print(compressed.shape)
		return compressed

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

class CharacterSetModel(object):
	def __init__(self, paths, samples = -1):
		self.models = dict()
		for code, path in paths.items():
			pass
		pass

	def build_model(self):
		pass

