function [train_data, test_data, n] = lena_patches(T)
    I=double(imread('spams-matlab/data/lena.png'))/255;
    % extract 8 x 8 patches
    X=im2col(I,[16 16],'sliding');
    X=X-repmat(mean(X),[size(X,1) 1]);
    X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

    % s - starting point for taking patches
    s = floor(size(X,2)/2);
    train_data=X(:,s:s+T); %floor(size(X,2)/2));

    test_data=X(:,s+T+100:s+2*T+100); %floor(size(X,2)/2):end);
    n = size(X,1);
