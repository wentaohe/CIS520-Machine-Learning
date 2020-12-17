Boxconstrains = [1,10,100,1000,10000,100000];
Boxconstrains_Err = zeros(6,1);

for idx = 1:numel(Boxconstrains)
    err = 0;
    Boxconstrain = Boxconstrains(idx);
    for i = 1:5
        train_data = ['Synthetic/CrossValidation/Fold',mat2str(i),'/cv-train.mat'];
        load(train_data);
        
        test_data = ['Synthetic/CrossValidation/Fold',mat2str(i),'/cv-test.mat'];
        load(test_data);
        
        SVMModel = fitcsvm(cv_train(:,1:2),cv_train(:,3),'BoxConstraint',Boxconstrain);
        
        labels= predict(SVMModel, cv_test(:,1:2));
        
        err = err + classification_error(labels, cv_test(:,3));
    end
    Boxconstrains_Err(idx,1) = err/5;
end

[Boxconstrains_min,column]=find(Boxconstrains_Err==min(min(Boxconstrains_Err)));

load('Synthetic/train.mat');
load('Synthetic/test.mat');

SVMModel = fitcsvm(train(:,1:2),train(:,3),'BoxConstraint',Boxconstrains(Boxconstrains_min));
train_labels = predict(SVMModel, train(:,1:2));
test_labels = predict(SVMModel, test(:,1:2));

classification_error(train_labels, train(:,3))  %train err
classification_error(test_labels, test(:,3))  %test err
Boxconstrains_Err(Boxconstrains_min,column)   %valid err

decision_boundary_SVM(test(:,1:2), test(:,3), SVMModel, 100, 'Linear')


