%Part a
load('MNIST_train.mat')
%last=X_train(12000,:);
%last_ex=reshape(last,[28 28]);
%imagesc(last_ex')
%colormap('gray')
%Part b
figure();
coef=pca(X_train);
first=reshape(coef(:,1),[28 28]);
imagesc(first')
colormap('gray')
figure()
second=reshape(coef(:,2),[28 28]);
imagesc(second')
colormap('gray')
figure()
third=reshape(coef(:,3),[28 28]);
imagesc(third')
colormap('gray')
%Part c
%for 1st, 2nd PC
 %zero=X_train(find(Y_train==1),:);
 %seven=X_train(find(Y_train==8),:);
 %[coef1, score1, ~]=pca(zero);
 %[coef2, score2, ~]=pca(seven);
 %scatter(score1(:,1),score1(:,2),'r');
 %xlabel('1st PC');
 %ylabel('2nd PC');
 %hold on
 %scatter(score2(:,1),score2(:,2),'b');
 %legend('Digit 0','Digit 7')
 %for 100th, 101st PC
 %figure()
 %scatter(score1(:,100),score1(:,101),'r');
 %xlabel('100th PC');
 %ylabel('101st PC');
 %hold on
 %scatter(score2(:,100),score2(:,101),'b');
 %legend('Digit 0','Digit 7')
%Part d
%[coeff,score,latent,tsquared,explained,mu] = pca(X_train);
% accuracy=[];
 %summ=0;
 %for i=1:784
  %   summ=summ+explained(i);
  %   accuracy(i)=summ;
 %end
 %plot(1:784, accuracy)
 %xlabel('Number of Principal Components');
 %ylabel('Accuracy');
 %idx=[];
 %for i=10:10:100
 %   [~,idx(i/10)]=min(abs(accuracy-i));
 %   if (accuracy(idx(i/10))-i)<0
 %       idx(i/10)=idx(i/10)+1;
 %   end
 %end
 %idx(10)=idx(10)-1;
%Part e
%[coeff,score,latent,tsquared,explained,mu] = pca(X_train);
 %kk=[500 6000 10000];
 %for k=kk
 %    figure()
 %for i=1:10
 %    subplot(3,4,i)
 %    xhat=score(:,1:idx(i))*coeff(:,1:idx(i))';
 %    example=reshape(xhat(k,:),[28 28]);
 %    imagesc(example')
 %    title(strcat(int2str(i*10),'% Reconstruction'))
 %    colormap('gray')   
 %end
 %subplot(3,4,i)
 %example=reshape(X_train(k,:),[28 28]);
 %imagesc(example')
 %colormap('gray')
 %title('Original Image')
 %end
%Part f
%X = bsxfun(@minus, X_train, mean(X_train));
 %autoenc = trainAutoencoder520(X, idx(6),'MaxEpochs',250,'LossFunction','mse');
 %new_weights=autoenc.EncoderWeights';
 %xhat=new_weights*coeff(:,1:idx(6))';
 %[~,~,~,~,explained_xhat,~] = pca(xhat);
 %accuracy_xhat=sum(explained_xhat(1:idx(6)));
%autoenc_linear = trainAutoencoder520(X, idx(2),'MaxEpochs',250,'LossFunction','mse','EncoderTransferFunction', 'purelin', 'DecoderTransferFunction', 'purelin');