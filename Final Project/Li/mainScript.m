% Load the data
data = load('training_data.mat');
X = data.train_inputs;
y = data.train_labels;
N = size(y);

% PCA to columns 22:2021
[coeff, score, latent, tsqured, explained, mu] = pca(X(:,22:2021));

% construct the accuracy matrix
nFeature = size(explained);
accuracyMat = zeros(nFeature);

accuracyMat(1) = explained(1);
for i = 2:nFeature
    accuracyMat(i) = accuracyMat(i-1) + explained(i); 
end
   
figure(1);
plot(1:nFeature, accuracyMat ./ 100);
xlabel('nFeatures');
ylabel('X% Explained');

n_90 = find(accuracyMat>90,1);

X_mean = mean(X(:,22:2021));
X_center = X(:,22:2021) - X_mean;

rng('default');
autoenc_90_linear = trainAutoencoder520(X_center', n_90, ...
                            'MaxEpochs', 500,...
                            'LossFunction', 'mse',...
                            'EncoderTransferFunction','purelin',...
                            'DecoderTransferFunction','purelin'...
                            );
XReconstructed_90_linear = predict(autoenc_90_linear, X_center');
mseError = mse(X_center-XReconstructed_90_linear');
Z = encode(autoenc_90_linear, X_center');
X_train = [X(:,1:21), Z'];

nfold = 5;
indices = crossvalind('Kfold', N(1), nfold);
nhyper = 5;
Lambda = logspace(-10,-1,nhyper);
predicted = zeros(N);

output = zeros(nfold, nhyper);
for n = 1:9
    for i = 1:nfold
        test = (indices == i);
        train = ~test;
        [Mdl, FitInfo] = fitrlinear(X_train(train,:),...
            y(train,1),...
            'Regularization', 'ridge',...
            'Learner','leastsquares',...
            'Lambda',Lambda);
        y_test = predict(Mdl, X_train(test,:));
        for j = 1:nhyper
            pre = y(test,:);
            pre(:,n) = y_test(:,j);
            output(i,j) = error_metric(pre, y(test,:));
        end
    end
    output = mean(output,1);
    best_hyper = find(output==min(output));
    [Mdl, FitInfo] = fitrlinear(X_train,...
            y(:,n),...
            'Regularization', 'ridge',...
            'Learner','leastsquares',...
            'Lambda',Lambda(best_hyper));
    predicted(:,n) = predict(Mdl, X_train);
end

pre_kernel = y;
for n = 1:9
    rng('default')
    [Mdl,FitInfo] = fitrkernel(X_train,y(:,n),...
        'Learner','leastsquares');
    pre_kernel(:,n) = predict(Mdl, X_train);
end

pre_svr = y;
for n = 1:9
    rng('default')
    Mdl = fitrsvm(X_train,y(:,n),'KernelFunction', 'gaussian');
    pre_svr(:,n) = predict(Mdl, X_train);
end
error_metric(pre_svr, y)