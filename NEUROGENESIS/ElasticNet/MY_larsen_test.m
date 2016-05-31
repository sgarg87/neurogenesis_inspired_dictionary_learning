% Example script for performing elastic net regression on the diabetes data
% set.

clear; close all; clc;

addpath('lib');
load diabetes

X = diabetes.x;

X = X(:,[1 2]);

X = [X X];


X = diabetes.x2;
X = normalize(X);
y = diabetes.y;
y = center(y);
[n p] = size(X);

b1 = lars(X, y, 'lasso', 0, 0, [], 1);
for i=1:p
    
    x = b1(i,:)';
    sol = x'
dual_cons=(abs(X'*(X*x-y)))'
end


%------------------
lambda2 = 0.1;
b1 = larsen(X, y, lambda2, 0, 1);

Xaug = (1+lambda2)^(-1/2)*[X ; sqrt(lambda2)*eye(p)]
yaug = [y; zeros(p,1)];

for i=1:p
    
    x = b1(i,:)'*(1+lambda2)^(-1/2);
    sol = x'
dual_cons=(abs(Xaug'*(Xaug*x-yaug)))'
end


t1 = sum(abs(b1),2)/max(sum(abs(b1), 2));

figure;
plot(t1, b1, '-');
title('Elastic net (LARS-EN)');
