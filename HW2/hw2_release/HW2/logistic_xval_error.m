function [error] = logistic_xval_error(X, Y, part)
% LOGISTIC_XVAL_ERROR - Logistic regression cross-validation error.
%
% Usage:
%
%   ERROR = logistic_xval_error(X, Y, PART)
%
% Returns the average N-fold cross validation error of the logistic regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, LOGISTIC_REGRESSION

% FILL IN YOUR CODE HERE

    n = max(part);
    total_error = zeros(n,1);
    
    for i=1:n
        Xtrain = X(part~=i,:);
        Ytrain = Y(part~=i,:);
        Xtest = X(part == i,:);
        Yactual = Y(part == i,:);    
        y_hat = logistic_regression(Xtrain,Ytrain,Xtest,.0002,500);
        totErrors = sum(Yactual ~= y_hat);
        total_error(i) = totErrors / length(y_hat);
    end
    
    error = mean(total_error);
end