function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

pred_labels=randn(size(test_inputs,1),size(train_labels,2));

X = train_inputs;
y = train_labels;
X_test = test_inputs;

% Step 1: shrink features using autoencoder
[coeff, score, latent, tsquared, explained, mu] = pca(X);
% construct the accuracy matrix
nFeature = size(explained);
accuracyMat = zeros(nFeature);

accuracyMat(1) = explained(1);
for i = 2:nFeature
    accuracyMat(i) = accuracyMat(i-1) + explained(i); 
end
n_90 = find(accuracyMat>90,1);

X_mean = mean(X(:,22:2021));
X_center = X(:,22:2021) - X_mean;

rng('default');
autoenc_90_linear = trainAutoencoder520(X_center', n_90, ...
                            'MaxEpochs', 200,...
                            'LossFunction', 'mse',...
                            'EncoderTransferFunction','purelin',...
                            'DecoderTransferFunction','purelin'...
                            );
XReconstructed_90_linear = predict(autoenc_90_linear, X_center');
mseError = mse(X_center-XReconstructed_90_linear');

% Encode train and test dataset
Z_train = encode(autoenc_90_linear, X_center');
X_test_center = X_test(:,22:2021) - X_mean;
Z_test = encode(autoenc_90_linear, X_test_center')

X_train = [X(:, 1:21) Z_train'];
X_test = [X_test(:, 1:21) Z_test'];

N = size(test_inputs);
pred_labels = zeros(N(1),9);
for n = 1:9
    rng('default')
    Mdl = fitrsvm(X_train,y(:,n),'KernelFunction', 'gaussian');
    pred_labels(:,n) = predict(Mdl, X_test);
end

end

