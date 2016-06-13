function [train_data, test_data, n] = lena_patches(T)
    I=double(imread('spams-matlab/data/lena.png'));
%     
    X=im2col(I,[16 16],'sliding');
%     X=im2col(I,[32 32],'sliding');
    %     
    X = preprocess_data(X);
    % 
    % s - starting point for taking patches
    s = floor(size(X,2)/2);
    train_data=X(:,s:s+T); %floor(size(X,2)/2));

    test_data=X(:,s+T+100:s+2*T+100); %floor(size(X,2)/2):end);
    n = size(X,1);
