from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from scipy import signal

k = 1
resampling_length = 40

DATA_FOLDER = "/preprocessed_sequences"
XTrain = []
YTrain = []
XTest = []
YTest = []

# PREPROCESSING & FEATURE EXTRACTION
# Iterate over all sample files
for filename in sorted(os.listdir(os.getcwd()+ DATA_FOLDER)):
    current_file = open(DATA_FOLDER[1:] + '/' + filename, 'r')

    # Train sample data
    if "trainimg" in filename and "inputdata" in filename:
        X = []
        lines = current_file.readlines()
        for line in lines:
            X.append([])
            currentLine = line.split()
            for num in currentLine:
                X[-1].append(float(num))
        X = np.array(X)
        X = np.transpose(X[1 :])
        newX = []
        for row in X:
            newX.append(signal.resample(row, resampling_length))
        newX = np.transpose(np.array(newX))
        XX = np.reshape(newX, np.prod(np.shape(newX)))
        XTrain.append(XX)

    # Train sample label
    elif "trainimg" in filename and "targetdata" in filename:
        line = current_file.readline().split()
        digit_class = 0
        for i in range(0, 10):
            if int(line[i]) == 1:
                digit_class = i
                break
        YTrain.append(digit_class)

    # Test sample data
    if "testimg" in filename and "inputdata" in filename:
        X = []
        lines = current_file.readlines()
        for line in lines:
            X.append([])
            currentLine = line.split()
            for num in currentLine:
                X[-1].append(float(num))
        X = np.array(X)
        X = np.transpose(X[1 :])
        newX = []
        for row in X:
            newX.append(signal.resample(row, resampling_length))
        newX = np.transpose(np.array(newX))
        XX = np.reshape(newX, np.prod(np.shape(newX)))
        XTest.append(XX)

    # Test sample label
    elif "testimg" in filename and "targetdata" in filename:
        line = current_file.readline().split()
        digit_class = 0
        for i in range(0, 10):
            if int(line[i]) == 1:
                digit_class = i
                break
        YTest.append(digit_class)

# Convert everything to numpy arrays
XTrain = np.array(XTrain)
YTrain = np.array(YTrain)
XTest = np.array(XTest)
YTest = np.array(YTest)

# CLASSIFICATION: Build and train model
kNN = KNeighborsClassifier(n_neighbors = k)
kNN.fit(XTrain, YTrain)
YHat = kNN.predict(XTest)

# POSTPROCESSING: Get predictions and calculate error rate
different_classes = YHat - YTest # If classes are the same, elements will be 0
errors = 0.0
for c in different_classes:
    if c != 0:
        errors += 1
error_rate = errors/len(YHat)
print("kNN Result: k = %d and Resampling Length = %d ==> Error Rate = %.2f%%" % (k, resampling_length, error_rate * 100))