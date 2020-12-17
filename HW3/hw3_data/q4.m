clear
load('test_data.mat');
load('test_y.mat');
load('train_data.mat');
load('train_y.mat');
n = 64;

X1 = X(:,1);
w1 = inv(X1'*X1)*X1'*Y; 
Err1 = (Y-X1*w1)'*(Y-X1*w1) 

X2 = X(:,1:2);
w2 = inv(X2'*X2)*X2'*Y; 
Err2 = (Y-X2*w2)'*(Y-X2*w2) 

w3 = inv(X'*X)*X'*Y; 
Err3 = (Y-X*w3)'*(Y-X*w3) 

Err_bits1 = n*log2(Err1/n) 
Err_bits2 = n*log2(Err2/n) 
Err_bits3 = n*log2(Err3/n) 

AIC_bits1 = Err_bits1+2*1 
AIC_bits2 = Err_bits2+2*2 
AIC_bits3 = Err_bits3+2*3 

BIC_bits1 = Err_bits1+2*.5*log2(64)*1 
BIC_bits2 = Err_bits2+2*.5*log2(64)*2 
BIC_bits3 = Err_bits3+2*.5*log2(64)*3 

Xtest1 = Xtest_new(:,1); 
test_Err1 = (Ytest_new-Xtest1*w1)'*(Ytest_new-Xtest1*w1) 
Xtest2 = Xtest_new(:,1:2);
test_Err2 = (Ytest_new-Xtest2*w2)'*(Ytest_new-Xtest2*w2) 
Xtest3 = Xtest_new;
test_Err3 = (Ytest_new-Xtest3*w3)'*(Ytest_new-Xtest3*w3) 
