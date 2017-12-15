# This file is a demo for a single sample

import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
from rnn import test

if __name__ == '__main__':
	# main()
	fnum = input("please enter one number: ")
	
	raw = np.loadtxt('./sequences/testimg-'+str(fnum)+'-targetdata.txt')
	label = raw[0,:10]
	values = raw[:,10:12]

	figx = [values[0,0]]
	figy = [values[0,1]]
	for i in range(1,len(values)):
		figx.append(figx[-1]+values[i,0])
		figy.append(figy[-1]+values[i,1])
	figy = [-i for i in figy]
	rnn_result = test.single_test(values)
	knn_result = -1 #your function
	svm_result = -1 #your function
	print "True Lable: " + str(label.argmax())
	print "RNN Lable: " + str(rnn_result)
	print "KNN Lable: " + str(knn_result)
	print "SVM Lable: " + str(svm_result)
	plt.axis([0,30,-30,0])
	plt.plot(figx,figy,'.')
	plt.show()