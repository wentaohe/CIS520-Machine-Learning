% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.

load('X_noisy.mat');
%load('X.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('Y.mat');
X = X_noisy;

N_folds = [3, 5, 9, 15];
errors_xval = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)


%2.1 3.3
for trial = 1:100
    
    % fill in the rest of your code here. 
    parts = make_xval_partition(600, 4);
    X_train = X(parts~=4,:);
    Y_train = Y(parts~=4,:);
    X_test = X(parts==4,:);
    Y_test = Y(parts==4,:);
    
    for i = 1:4
        
        part = make_xval_partition(450, N_folds(i));
        %Plot 2.1.1 & Plot 2.2.2
        %errors_xval(trial, i) = knn_xval_error(X_train, Y_train, 1, part, 'l2');
        %labels = k_nearest_neighbours(X_train,Y_train,X_test,1,'l2');
        
        %Plot 2.1.3 & Plot 2.1.4
        %errors_xval(trial,i) = kernreg_xval_error(X_train, Y_train, 1, part);
        %labels = kernel_regression(X_train,Y_train,X_test,1);
        
        %Plot 3.3.1 % Plot 3.3.2
        errors_xval(trial,i) = logistic_xval_error(X_train, Y_train, part);
        labels = logistic_regression(X_train,Y_train,X_test,.0002,500);

        testError = sum(Y_test ~= labels) / length(labels);
        errors_test(trial, i) = testError;
    end
end

%2.2
%{
k = [1, 3, 4, 6, 9, 14, 22, 25];
sigma = [1, 3, 5, 7, 9, 11];

for trial = 1:100
    
    % fill in the rest of your code here. 
    parts = make_xval_partition(600, 4);
    X_train = X(parts~=4,:);
    Y_train = Y(parts~=4,:);
    X_test = X(parts==4,:);
    Y_test = Y(parts==4,:);
    
    for i = 1:8 %KNN
    %for i = 1:6 %Kernel regression   
        part = make_xval_partition(450,10);
        %Plot 2.1.1 & Plot 2.2.2
        errors_xval(trial, i) = knn_xval_error(X_train, Y_train, k(i), part, 'l2');
        labels = k_nearest_neighbours(X_train, Y_train, X_test, k(i), 'l2');
        
        %Plot 2.1.3 & Plot 2.1.4
        %errors_xval(trial,i) = kernreg_xval_error(X_train, Y_train, sigma(i), part);
        %labels = kernel_regression(X_train,Y_train,X_test,sigma(i));
        
        testError = sum(Y_test ~= labels) / length(labels);
        errors_test(trial, i) = testError;
    end
end
%}

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval); e = std(errors_xval); x = [3, 5, 9, 15];
%y = mean(errors_xval); e = std(errors_xval); x = [1, 3, 4, 6, 9, 14, 22, 25]; % <- computes mean across all trials
%y = mean(errors_xval); e = std(errors_xval); x = [1, 3, 5, 7, 9, 11];
errorbar(x, y, e);

hold on;
y = mean(errors_test); e = std(errors_test); x = [3, 5, 9, 15];
%y = mean(errors_test); e = std(errors_test); x = [1, 3, 4, 6, 9, 14, 22, 25]; % <- computes mean across all trials
%y = mean(errors_test); e = std(errors_test); x = [1, 3, 5, 7, 9, 11];
errorbar(x, y, e);

%Plot 2.1.1 & 2.2.1
%title('Original data, N = [3, 5, 9, 15] with KNN, K = 1'); 
%title('Original data, N = 10 with KNN, K \in [1,3,4,6,9,14,22,35]'); 

%Plot 2.1.2 & 2.2.2
%title('Noisy data, N = [3, 5, 9, 15] with KNN, K = 1');
%title('Noisy data, N = 10 with KNN, K \in [1,3,4,6,9,14,22,35]'); 

%Plot 2.1.3 & 2.2.3
%title('Original data, N = [3, 5, 9, 15] with Kernel Regression, Sigma = 1');
%title('Original data, N = 10 with Kernel Regression, Sigma \in [1, 3, 5, 7, 9, 11]');

%Plot 2.1.4 & 2.2.4
%title('Noisy data, N = [3, 5, 9, 15] with Kernel Regression, Sigma = 1');
%title('Noisy data, N = 10 with Kernel Regression, Sigma \in [1, 3, 5, 7, 9, 11]');

%Plot 3.3
%title('Original data, N = [3, 5, 9, 15] with Logistic Regression, Step Size = 500');

xlabel('N');
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;