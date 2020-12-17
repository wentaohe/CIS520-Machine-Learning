x_pca = [2,4,6,9,15,25,43,88,264];
y_pca = [0.1451,0.2457,0.3260,0.4044,0.5051,0.6058,0.7030,0.8002,0.9002]*100;
x_auto_log = [6,25,264];
y_auto_log = [0.2461,0.4719,0.8490]*100;
x_auto_lin = [6,25,264];
y_auto_lin = [0.3160,0.5987,0.8504]*100;
plot(x_pca,y_pca);
hold on
plot(x_auto_log, y_auto_log);
hold on
plot(x_auto_lin, y_auto_lin);
legend('pca', 'nonlin', 'lin');
title('Reconstruction Accuracy')