import tensorflow as tf
from model import DigitsModel

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

digit_model = DigitsModel(x_train, y_train, iters = 10)
results = digit_model.evaluate(x_test, y_test)

print(results)