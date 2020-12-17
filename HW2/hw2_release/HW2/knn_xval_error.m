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

n = max(part);
total_error = zeros(n,1);

for i=1:n
    Xtrain = X(part~=i,:);
    Ytrain = Y(part~=i,:);
    Xtest = X(part == i,:);   
    y_hat = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distFunc);
    totErrors = sum(Y(part == i,:) ~= y_hat);
    total_error(i) = totErrors / length(y_hat);    
end

error = mean(total_error);