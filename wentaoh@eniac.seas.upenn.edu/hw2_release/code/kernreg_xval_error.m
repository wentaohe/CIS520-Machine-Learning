function [error] = kernreg_xval_error(X, Y, sigma, part)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

% FILL IN YOUR CODE HERE
if (nargin < 5)
    distFunc='l2';
end

n = max(part);

total_error = 0;

for i = 1:n
    XtestIndex = find(part(find(X(:,1))) == i);
    Xtest = X((part(find(X(:,1))) == i),:);
    testTrueLabels = Y(ismember(find(Y), XtestIndex) > 0);
    
    trainPointIndex = find(part(find(X(:,1))) ~= i);
    Xtrain = X((part(find(X(:,1))) ~= i),:);
    Ytrain = Y(ismember(find(Y), trainPointIndex) > 0);
    
    test_kernreg_labels = kernel_regression(Xtrain, Ytrain, Xtest, sigma);
    fold_error = sum(testTrueLabels.* test_kernreg_labels <= 0);
    fold_error = fold_error/size(testTrueLabels,1);
    total_error = total_error + fold_error;
end

total_error = total_error/n;