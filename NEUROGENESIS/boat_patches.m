function [data0, test_data0] = boat_patches(T)
    I=double(imread('spams-matlab/data/boat.png'))/255;
    % extract 8 x 8 patches or 16x 16
    X=im2col(I,[16 16],'sliding');
    X=X-repmat(mean(X),[size(X,1) 1]);
    X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
    % s - starting point for taking patches
    s = floor(size(X,2)/2);

    data0=X(:,s:s+T);   % this will be 'nonstationary' regime, i.e. when testing on patches from new image, 
                      % different from training and test image                  
    test_data0 = X(:,s+T+100:s+2*T+100);
