%
%   simple example to show the usage of l1_ls
%

% problem data
X  = [1    0    0   0.5;...\
      0    1  0.2   0.3;...\
      0  0.1    1   0.2];
b0 = [1 0 1 0 0 0 0 0]';    % original signal
X = [ X X];

y  = X*b0;          % measurements with no noise
lambda = 0.01;      % regularization parameter
rel_tol = 0.0001;     % relative target duality gap

[x,status]=l1_ls(X,y,lambda,rel_tol)

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

b1 = lars(X, y, 'lasso', t);