clc;
clear;

load('Breast-Cancer/trainingdata.mat');
load('Breast-Cancer/testdata.mat');

Cs = [0,0.0001,0.001,0.01,0.1,1];
Cs_Err = zeros(6,1);

for index = 1:size(Cs, 2)        
    C = Cs(index);
    net = feedforwardnet([10]);
    net.layers{1}.transferFcn = 'poslin';
    %net.layers{1}.transferFcn = 'logsig';
    net.performFcn = 'crossentropy';
    %net.performFcn = 'mse';
    net.performParam.regularization = C;
    net.trainFcn = 'trainrp';
    
    err = 0;
    for i = 1:5
        train_data = ['Breast-Cancer/CrossValidation/Fold',mat2str(i),'/cv-train.mat'];
        load(train_data);

        test_data = ['Breast-Cancer/CrossValidation/Fold',mat2str(i),'/cv-test.mat'];
        load(test_data);

        net = train(net,cv_train(:,1:9)',cv_train(:,10)');
        labels= net(cv_test(:,1:9)');

        err = err + classification_error(sign(labels), cv_test(:,10)');
    end
    Cs_Err(index,1) = err/5;
    
end

[Cs_min,column]=find(Cs_Err==min(min(Cs_Err)));

if length(Cs_min) > 1
    Cs_min = Cs_min(1);
    column = column(1);
end

Best_c = Cs(Cs_min)
val_error = Cs_Err(Cs_min,column)

net = feedforwardnet(10);
net.layers{1}.transferFcn = 'poslin';
%net.layers{1}.transferFcn = 'logsig';
net.performFcn = 'crossentropy';
%net.performFcn = 'mse';
net.performParam.regularization = Cs(Cs_min);
net.trainFcn = 'trainrp';

net = train(net,train_inputs',train_labels');     
labels= net(test_inputs')';
labels = sign(labels);
save('NNlabels.mat','labels');

net = train(net,train_inputs',train_labels');     
labels= net(train_inputs')';
labels = sign(labels);
train_error = classification_error(sign(labels), train_labels)
