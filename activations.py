from tensorflow import multiply
from keras.backend import sigmoid

def gated_linear(x):
	return multiply(x, sigmoid(x))