% Example script for performing LASSO regression on the diabetes data
% set.

clear; close all; clc;

addpath('lib');
load diabetes

X = diabetes.x;
X = normalize(X);

X = [X X];

y = diabetes.y;
y = center(y);
[n p] = size(X);

lambda = 1000;      % regularization parameter
rel_tol = 0.00001;     % relative target duality gap

[x,status]=l1_ls(X,y,lambda,rel_tol);

 dual_cons=(abs(X'*(X*x-y)))'
 
 % Scaled Norm Solvers
 
[LassoBlockCoordinate(X,y,lambda) ...
    LassoConstrained(X,y,lambda,'mode',2) ...
    LassoGaussSeidel(X,y,lambda) ...
    LassoGrafting(X,y,lambda) ...
    LassoIteratedRidge(X,y,lambda) ...
    LassoNonNegativeSquared(X,y,lambda) ...
    LassoPrimalDualLogBarrier(X,y,lambda) ...
    LassoProjection(X,y,lambda) ...
    LassoShooting(X,y,lambda) ...
    LassoSubGradient(X,y,lambda) ...
    LassoUnconstrainedApx(X,y,lambda) ...
    LassoUnconstrainedApx(X,y,lambda,'mode2',1)]
pause;
 
 
 

for i=1:size(b1,1)
    dual_cons=(abs(X'*(X*b1(i,:)'-y)))'
    b1(i,:)
end

[s_opt, b_opt, res_mean, res_std] = crossvalidate(@lars, 10, 1000, X, y, 'lasso', 0, 0, [], 0);
cvplot(s_opt, res_mean, res_std);


figure;
hold on;
plot(s1, b1, '.-');
axis tight;
ax = axis;
line([s_opt s_opt], [ax(3) ax(4)], 'Color', 'r', 'LineStyle', '-.');
legend
title('Least Angle Regression (LASSO)');

%%%%%
aa1=(X(:,3)'*y)/(X(:,3)'*X(:,3));
aa2=(X(:,9)'*y)/(X(:,9)'*X(:,9));

bb1 = [0 0 aa1 0 0 0 0 0 0 0];
bb2 = [0 0 0   0 0 0 0 0 aa2 0];

err1 = sum( (X*bb1'-y).^2)
err2 = sum( (X*bb2'-y).^2)
err = sum( (X*b1(2,:)'-y).^2 )