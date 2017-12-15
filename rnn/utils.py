import numpy as np
import os
import pickle
import tensorflow as tf
from scipy import signal
import model


tf.app.flags.DEFINE_string('data_path', './sequences', 'orginal data path')
tf.app.flags.DEFINE_string('dataset_path', './rnn', 'processed dataset path')
tf.app.flags.DEFINE_string('model_path', './rnn/trained', 'saved model path')
tf.app.flags.DEFINE_boolean('resample', True, 'resample data to same size')

tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size to use during training.')
tf.app.flags.DEFINE_integer('timesteps', 20, 'Lenght of Sequences to process.')
tf.app.flags.DEFINE_integer('num_hidden', 30, 'Number of units in each layer.')
tf.app.flags.DEFINE_integer('num_layer', 3, 'Number of Layers.')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of Classes.')
tf.app.flags.DEFINE_integer('num_dimension', 2, 'Dimension of input.')
tf.app.flags.DEFINE_integer('training_epoch', 10, 'Batch size to use during training.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')


FLAGS = tf.app.flags.FLAGS

#reading the data and resample them if needs.
def read_data(training = True, resample = True):
	labels = []
	data = []
	if training:
		mode = 'trainimg'
		fname = 'trainset.dat'
	else:
		mode = 'testimg'
		fname = 'testset.dat'
	if resample:
		fname = 'resample-'+ str(FLAGS.timesteps) + '-' + fname

	if os.path.isfile(FLAGS.dataset_path+'/' + fname):
		with open(FLAGS.dataset_path + '/' + fname,'r')as f:
			data = pickle.load(f)
			labels = pickle.load(f)
		return data, labels

	for f in os.listdir(FLAGS.data_path):
		if ('targetdata' in f) and (mode in f):
			raw = np.loadtxt(FLAGS.data_path+'/'+f)
			labels.append(raw[0,:10])
			values = raw[:,10:12]
			if resample:
				values = signal.resample(values[1:],FLAGS.timesteps)
				
			data.append(values)

	with open(FLAGS.dataset_path + '/' + fname,'w') as f:
		pickle.dump(data,f)
		pickle.dump(labels,f)
	return data,labels

#return an iterator for batches
def next_batch(all_data,all_labels, bs):
	batch_size = bs
	timesteps = FLAGS.timesteps
	num_batch = len(all_data) / batch_size
	for i in range(num_batch-1):
		start = i*batch_size
		end = (i+1)*batch_size
		b_x = all_data[start:end]
		b_y = all_labels[start:end]
		for i in range(len(b_x)):
			b_x[i] = b_x[i][1:]
			if len(b_x[i]) >= timesteps:
				b_x[i] = b_x[i][:timesteps]
			else:
				dif = timesteps - len(b_x[i])
				b_x[i] = np.append(b_x[i],[[0,0]]*dif,axis=0)
		b_x = np.array(b_x)
		b_y = np.array(b_y)

		yield [b_x,b_y]
