import numpy as np
import os
from scipy import signal

NORMALIZED_LENGTH = 40
DATA_FOLDER = "/sequences"
PREPROCESSED_FOLDER = "/preprocessed_sequences"

for filename in os.listdir(os.getcwd()+ DATA_FOLDER):
    current_file = open(DATA_FOLDER[1 :] + '/' + filename, 'r')

    if "inputdata" in filename:
        X = []
        lines = current_file.readlines()
        for line in lines:
            X.append([])
            currentLine = line.split()
            for num in currentLine:
                X[-1].append(float(num))
        X = np.array(X)

        X = np.transpose(X)
        newX = []
        for row in X:
            newX.append(signal.resample(row, NORMALIZED_LENGTH))
        newX = np.transpose(np.array(newX))

        with open(PREPROCESSED_FOLDER[1 :] + '/' + filename, 'w') as file:
            for row in newX:
                file.write(str(row[0]))
                for element in row[1 :]:
                    file.write(' ' + str(element))
                file.write('\n')

    if "targetdata" in filename:
        line = current_file.readline().split()
        with open(PREPROCESSED_FOLDER[1:] + '/' + filename, 'w') as file:
            file.write(str(line[0]))
            for i in range(1, 10):
                file.write(' ' + str(line[i]))