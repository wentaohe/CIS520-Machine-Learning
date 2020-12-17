x1=0:0.01:pi;
y1=sqrt(2215-x1.^2);
plot(x1,y1)
hold on
x2=0:0.01:pi/2;
y2=x2.*tan(x2);
plot(x2,y2)
hold on
x3=pi/2:0.01:pi;
y3=-x3.*cot(x3);
plot(x3,y3);
axis([1 3.5 46.5 47.5])

x_intersect = 0;
