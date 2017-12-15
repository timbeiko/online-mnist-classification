import numpy as np
import os
import pickle
import tensorflow as tf
from scipy import signal

FLAGS = tf.app.flags.FLAGS
#define the model
def RNN(x):

	weights = {
		'out': tf.Variable(tf.random_normal([FLAGS.num_hidden, FLAGS.num_classes])),
		'in': tf.Variable(tf.random_normal([FLAGS.num_dimension, FLAGS.num_hidden]))
		}
	biases = {
		'out': tf.Variable(tf.random_normal([FLAGS.num_classes])),
		'in': tf.Variable(tf.random_normal([FLAGS.num_hidden]))
		}

	layers = []
	#define rnn layers
	for i in range(FLAGS.num_layer):
		layer = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden)
		layer = tf.contrib.rnn.DropoutWrapper(layer)
		layers.append(layer)
	#stack layers
	rnn = tf.contrib.rnn.MultiRNNCell(layers)

	outputs, states = tf.nn.dynamic_rnn(rnn, x, dtype=tf.float32)

	outputs = tf.transpose(outputs, [1, 0, 2])
	last = outputs[-1]
	#add one fully connected layer (prediction layer) to the model
	logits = tf.matmul(last, weights['out']) + biases['out']

	return logits
