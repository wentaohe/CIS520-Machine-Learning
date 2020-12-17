function [error] = knn_xval_error(X, Y, K, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(X, Y, K, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART), corresponding to the number of folds.
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, K_NEAREST_NEIGHBOURS

% FILL IN YOUR CODE HERE
if (nargin < 5)
    distFunc='l2';
end

n = max(part);

total_error = 0;

for i = 1:n
    XtestIndex = find(part(find(X(:,1))) == i);
    Xtest = X((part(find(X(:,1))) == i),:);
    testTrueLabels = Y(ismember(find(Y), XtestIndices) > 0);
    
    trainPointIndex = find(part(find(X(:,1))) ~= i);
    Xtrain = X((part(find(X(:,1))) ~= i),:);
    Ytrain = Y(ismember(find(Y), trainPointIndex) > 0);
    
    test_knn_labels = k_nearest_neighbours(Xtrain, Ytrain, K, Xtest, distFunc);
    
    fold_error = sum(testTrueLabels.* test_knn_labels < 0);
    fold_error = fold_error/size(testTrueLabels,1);
    total_error = total_error + fold_error;
end

error = total_error/n;