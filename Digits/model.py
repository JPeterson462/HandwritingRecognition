import tensorflow as tf

class DigitsModel(object):
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
			x_train = x_train / 255.0
		self.model.fit(x_train, y_train, epochs = iters)
	def evaluate(self, x_test, y_test, normalize = True):
		if normalize:
			x_test = x_test / 255.0
		return self.model.evaluate(x_test, y_test, return_dict=True)

