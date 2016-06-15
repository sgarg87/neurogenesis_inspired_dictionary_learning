function [train_data, test_data, n] = lena_patches(T, dir_path, patch_size)
    image_file_path = strcat(dir_path, 'spams-matlab/data/lena.png');
    % 
    I=double(imread(image_file_path));
    %     
    X=im2col(I,patch_size,'sliding');
    %     
    X = preprocess_data(X);
    % 
    % s - starting point for taking patches
    s = floor(size(X,2)/2);
    train_data=X(:,s:s+T); %floor(size(X,2)/2));

    test_data=X(:,s+T+100:s+2*T+100); %floor(size(X,2)/2):end);
    n = size(X,1);
end
