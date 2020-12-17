load('X_noisy.mat');
%load('X.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('Y.mat');
X = X_noisy;

X = X(1:450, :);
Y = Y(1:450, :);

%[~,err_decay] = gradient_ascent_decay(X,Y,0.0001,500);
[~,err_fixed] = gradient_ascent_fixed(X,Y,0.0001,500);

%y = err_decay;
y = err_fixed;
x = 1:500;
plot(x, y);

hold on
extra = ones(size(X, 1), 1);
X = [X extra];
[~,err_fixed] = gradient_ascent_fixed(X,Y,0.0001,500);

%y = err_decay;
y = err_fixed;
plot(x, y);

%title('Error evolution : Original Data');
title('Error evolution : Noisy Data');

xlabel('Iteration');
ylabel('Error');
%legend('Decaying step size','Fixed step size');
legend('Fixed','Fixed with extra feature');
hold off