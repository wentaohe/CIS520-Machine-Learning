function [error] = kernreg_xval_error(X, Y, sigma, part)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART, DISTFUNC)
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

n = max(part);
total_error = zeros(n,1);

for i=1:n
    Xtrain = X(part~=i,:);
    Ytrain = Y(part~=i,:);
    Xtest = X(part == i,:);   
    y_hat = kernel_regression(Xtrain,Ytrain,Xtest,sigma);
    totErrors = sum(Y(part == i,:) ~= y_hat);
    total_error(i) = totErrors / length(y_hat);
end

error = mean(total_error);
    
