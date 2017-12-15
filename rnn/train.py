import numpy as np
import os
import pickle
import tensorflow as tf
from scipy import signal
import model
import utils



FLAGS = tf.app.flags.FLAGS


#train the model
def train():
	#define placeholders for batch
	X = tf.placeholder("float", [None, None, FLAGS.num_dimension]) 
	Y = tf.placeholder("float", [None, FLAGS.num_classes])
	#define the model and add to computational graph
	logits = model.RNN(X)
	#reading data using utils
	data, labels = utils.read_data(resample = FLAGS.resample)
	#add optimizer to computational graph
	prediction = tf.nn.softmax(logits)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
	train_op = optimizer.minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()
	batch_size = FLAGS.batch_size
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		#loop for each epoch
		for epoch in range(FLAGS.training_epoch):
			#loop on each batch
			for batch_x, batch_y in utils.next_batch(data,labels,batch_size):
				#prepare the batch for feeding
				batch_x = batch_x.reshape((batch_size, FLAGS.timesteps, FLAGS.num_dimension))
				#feeding the batch and optimze the model on the batch
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
			#compute the loss and accuracy for last batch of the epoch
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
			print("Step " + str(epoch) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
			
			# if (batch_size < 512)and(epoch%2 == 0):
			# 	batch_size *= 2
		#compute accuracy on test set
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

		save_path = saver.save(sess, FLAGS.model_path + "/model.ckpt")



if __name__ == '__main__':
	train()