from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import os 

# ============ PREPROCESSING ============
# Need to create set of files with normalized length
NORMALIZED_LENGTH = 40 # 10 for 'sequences', '40' for 'preprocessed_sequences'
DATA_FOLDER = "/preprocessed_sequences"
XTrain = []
YTrain = []
XTest = []
YTest = []

print "Loading input to memory"
for filename in os.listdir(os.getcwd() +  DATA_FOLDER):
    current_file = open(DATA_FOLDER[1 :] + '/' + filename, 'r')
    length_count = 0 

    # Training Input
    if "trainimg" in filename and "inputdata" in filename:
        X = []  
        while length_count < NORMALIZED_LENGTH:
            line = current_file.readline().split()
            length_count += 1
            X += map(float, line)
        X = np.array(X)
        XTrain.append(X)

    # Training Output
    elif "trainimg" in filename and "targetdata" in filename:
        line = current_file.readline().split()
        digit_class = 0 
        for i in range(0, 10):
            if int(line[i]) == 1:
                digit_class = i
                break
        YTrain.append(digit_class)


    # Testing Input
    elif "testimg" in filename and "inputdata" in filename:
        X = []  
        while length_count < NORMALIZED_LENGTH:
            line = current_file.readline().split()
            length_count += 1
            X += map(float, line)
        X = np.array(X)
        XTest.append(X)

    # Testing Output 
    elif "testimg" in filename and "targetdata" in filename:
        line = current_file.readline().split()
        digit_class = 0 
        for i in range(0, 10):
            if int(line[i]) == 1:
                digit_class = i
                break
        YTest.append(digit_class)

# Convert to numpy arrays 
XTrain = np.array(XTrain)
YTrain = np.array(YTrain)
XTest = np.array(XTest)
YTest = np.array(YTest)

# =======================================

# ========= FEATURE EXTRACTION ==========

# Finding best value for K, done below
for k in range(1, 11):
    # ============== TRAINING ===============

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(XTrain, YTrain)

    # =======================================

    # =========== CLASSIFICATION ============
    YHat = knn.predict(XTest)
    different_classes = YHat - YTest # If classes are the same, elements will be 0
    errors = 0.0
    for c in different_classes:
        if c != 0:
            errors += 1
    error_rate = errors/len(YHat)
    print "Number of neighbors: " + str(k) + " Error rate: " + str(error_rate)
    # =======================================

# =======================================