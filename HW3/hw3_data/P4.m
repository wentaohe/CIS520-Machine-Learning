clc;
clear;

load('test_data.mat');
load('test_y.mat');
load('train_data.mat');
load('train_y.mat');

w_a = X(:, 1:1)\Y;
w_b = X(:, 1:2)\Y;
w_c = X(:, 1:3)\Y;

% 1a
err1 = sum((X(:, 1:1)*w_a - Y).^2)
err2 = sum((X(:, 1:2)*w_b - Y).^2)
err3 = sum((X(:, 1:3)*w_c - Y).^2)

% 1b
N = size(X, 1);
err_bits1 = N * log2(err1/N)
err_bits2 = N * log2(err2/N)
err_bits3 = N * log2(err3/N)

% 1c
AIC_bits1 = err_bits1 + 2 * 1
AIC_bits2 = err_bits2 + 2 * 2
AIC_bits3 = err_bits3 + 2 * 3

% 1d
BIC_bits1 = err_bits1 + 2 * 0.5 * log2(N) * 1
BIC_bits2 = err_bits2 + 2 * 0.5 * log2(N) * 2
BIC_bits3 = err_bits3 + 2 * 0.5 * log2(N) * 3

% 2
% a) AIC: Model2
% b) BIC: Model2

% 3
% Yes
test_err1 = sum((Xtest_new(:, 1:1)*w_a - Ytest_new).^2)
test_err2 = sum((Xtest_new(:, 1:2)*w_b - Ytest_new).^2)
test_err3 = sum((Xtest_new(:, 1:3)*w_c - Ytest_new).^2)
