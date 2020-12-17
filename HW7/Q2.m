load('./data_new1.mat')
model = fitctree(Xtrain_full,Ytrain);
T_sim = predict(model, Xtest_full);
index = find(T_sim ~= Ytest);
Full_accuracy = 1-length(index)/length(Ytest)

load('./data_new1.mat')
Xtrain_random = fillmissing(Xtrain_random,'movmean',450);
model = fitctree(Xtrain_random,Ytrain);
Xtest_random = fillmissing(Xtest_random,'movmean',450);
T_sim = predict(model, Xtest_random);
index = find(T_sim ~= Ytest);
Rand_mean_accuracy = 1-length(index)/length(Ytest)

load('./data_new1.mat')
TF = ismissing(Xtrain_random)+0;
Xtrain_random = cat(2, Xtrain_random, TF); 
model = fitctree(Xtrain_random,Ytrain);
TF = ismissing(Xtest_random)+0;
Xtest_random = cat(2, Xtest_random, TF); 
T_sim = predict(model, Xtest_random);
index = find(T_sim ~= Ytest);
Rand_indicator_accuracy = 1-length(index)/length(Ytest)

load('./data_new1.mat')
Xtrain_nrandom = fillmissing(Xtrain_nrandom,'movmean',450);
model = fitctree(Xtrain_nrandom,Ytrain);
Xtest_nrandom = fillmissing(Xtest_nrandom,'movmean',450);
T_sim = predict(model, Xtest_nrandom);
index = find(T_sim ~= Ytest);
nRand_mean_accuracy = 1-length(index)/length(Ytest)

load('./data_new1.mat')
TF = ismissing(Xtrain_nrandom)+0;
Xtrain_nrandom = cat(2, Xtrain_nrandom, TF); 
model = fitctree(Xtrain_nrandom,Ytrain);
TF = ismissing(Xtest_nrandom)+0;
Xtest_nrandom = cat(2, Xtest_nrandom, TF); 
T_sim = predict(model, Xtest_nrandom);
index = find(T_sim ~= Ytest);
nRand_indicator_accuracy = 1-length(index)/length(Ytest)