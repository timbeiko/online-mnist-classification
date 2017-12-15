import numpy as np
import os
import pickle
import tensorflow as tf
from scipy import signal
import model
import utils



FLAGS = tf.app.flags.FLAGS
#feed one sample and predict the label
def single_test(x_data):
	values = np.array([signal.resample(x_data[1:],FLAGS.timesteps)])
	X = tf.placeholder("float", [None, None, FLAGS.num_dimension])
	Y = tf.placeholder("float", [None, FLAGS.num_classes])
	
	logits = model.RNN(X)
	# data, labels = utils.read_data(resample = FLAGS.resample)

	prediction = tf.nn.softmax(logits)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
	train_op = optimizer.minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()


	with tf.Session() as sess:

		saver.restore(sess, FLAGS.model_path +"/model.ckpt")
		print("Model restored.")
		res = sess.run([prediction],feed_dict={X:values})

	return res[0].argmax()
#feed all test set and compute the accuracy
def test():
	X = tf.placeholder("float", [None, None, FLAGS.num_dimension])
	Y = tf.placeholder("float", [None, FLAGS.num_classes])
	
	logits = model.RNN(X)
	data, labels = utils.read_data(resample = FLAGS.resample)

	prediction = tf.nn.softmax(logits)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
	train_op = optimizer.minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()


	with tf.Session() as sess:

		saver.restore(sess, FLAGS.model_path +"/model.ckpt")
		print("Model restored.")

		count = 0.0
		acc_sum = 0
		data_test,labels_test = utils.read_data(training = False, resample = FLAGS.resample)
		error = {}
		for i in range(FLAGS.num_classes):
			error[i] = 0
		for batch_x, batch_y in utils.next_batch(data_test,labels_test, 512):
			res = sess.run([prediction],feed_dict={X:batch_x})
			res = np.array(res[0])

			assert len(res) == len(batch_y) , 'not same size'
			for i in range(len(res)):
				if res[i].argmax() != batch_y[i].argmax():
					error[batch_y[i].argmax()] += 1

			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
			acc_sum += acc
			count += 1

		print ("accuracy on test: " + "{:.3f}".format(acc_sum/count))


		for i in error:
			print (i,error[i])

if __name__ == '__main__':
	test()