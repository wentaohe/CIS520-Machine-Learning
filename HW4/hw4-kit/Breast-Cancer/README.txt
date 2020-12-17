This folder contains the breast cancer classification data set drawn from the UCI Machine Learning repository (http://archive.ics.uci.edu/ml/datasets/breast+cancer). In the data files provided, each row is a separate training example.

Important Note: the labels are of type double (+1/-1), unlike HW2.

Files:
trainingdata.mat - Training instances + labels (600 x 9 + 600 x 1)
testdata.mat - Test instances (83 x 9)

Folders:
CrossValidation - contains the cross-validation data for each fold of the 5-fold cross-validation procedure on the training set. 

The cross-validation data has been generated using the following procedure: we divided the training data into 5 equal parts. For each fold cv_train contains 4 of the 5 parts and cv_test contains 1 left-out part. 

For doing cross-validation, you take each fold and train on the cv_train, and the error for the fold would be calculated using cv_test. Select the parameters with the least average cross-validation error.

Note: You are given test data for this dataset. You will have to upload your predictions for test data in the autograder where we will test your accuracy.

