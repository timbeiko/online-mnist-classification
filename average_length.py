import numpy as np
import os
from scipy import signal

NORMALIZED_LENGTH = 40
DATA_FOLDER = "/sequences"
PREPROCESSED_FOLDER = "/preprocessed_sequences"

filecount = 0.0
total_length = 0

for filename in os.listdir(os.getcwd()+ DATA_FOLDER):
    current_file = open(DATA_FOLDER[1 :] + '/' + filename, 'r')

    if "inputdata" in filename:
        filecount += 1
        lines = current_file.readlines()
        total_length += len(lines)

print total_length/filecount