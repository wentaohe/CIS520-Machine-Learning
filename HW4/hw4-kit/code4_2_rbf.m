clc;
clear;

load('Breast-Cancer/trainingdata.mat');
load('Breast-Cancer/testdata.mat');

sigmas = [0.1,1,10,100,1000];
Boxconstrains = [1,10,100,1000,10000,100000];
Error = zeros(6,1);
train_err = zeros(5,1);
val_err = zeros(5,1);
Best_C = [];

for index = 1:size(sigmas, 2)
    sigma = sigmas(index)
    Error = zeros(6,1);
    for idx = 1:numel(Boxconstrains)
        err = 0;
        Boxconstrain = Boxconstrains(idx);
        for i = 1:5
            train_data = ['Breast-Cancer/CrossValidation/Fold',mat2str(i),'/cv-train.mat'];
            load(train_data);

            test_data = ['Breast-Cancer/CrossValidation/Fold',mat2str(i),'/cv-test.mat'];
            load(test_data);

            SVMModel = fitcsvm(cv_train(:,1:9),cv_train(:,10),'BoxConstraint',...
                Boxconstrain,'KernelFunction','rbf','KernelScale',sigma);

            labels= predict(SVMModel, cv_test(:,1:9));
            err = err + classification_error(labels, cv_test(:,10));
        end
        Error(idx,1) = err/5;
    end
    
    [Boxconstrains_min,column]=find(Error==min(min(Error)));
    if length(Boxconstrains_min) > 1
        Boxconstrains_min = Boxconstrains_min(1);
        column = column(1);
    end
    
    C = Boxconstrains(Boxconstrains_min)
    Best_C(end+1) = C;
    
    SVMModel = fitcsvm(train_inputs,train_labels,'BoxConstraint',...
        Boxconstrains(Boxconstrains_min),'KernelFunction','RBF','KernelScale',sigma);

    train_err(index,1) = classification_error(predict(SVMModel, train_inputs), train_labels)
    val_err(index,1) = Error(Boxconstrains_min,column)
    
end   
sigmas = sigmas';
Best_C = Best_C';
table(sigmas, Best_C, train_err, val_err)
  
SVMModel = fitcsvm(train_inputs,train_labels,'BoxConstraint',10,'KernelFunction','RBF',...
    'KernelScale',100);
labels = predict(SVMModel, test_inputs);
labels = sign(labels);
save('SVMlabels.mat','labels');
  