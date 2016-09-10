function [data0, test_data0] = boat_patches(T, dir_path, patch_size)
    image_file_path = strcat(dir_path, 'spams-matlab/data/boat.png');
    %     
    I=double(imread(image_file_path));
    %     
    X=im2col(I,patch_size,'sliding');
    %
    X = double(X)/255;
    X = preprocess_data(X);
    %     
    % s - starting point for taking patches
    s = floor(size(X,2)/2);
    % this will be 'nonstationary' regime, i.e. when testing on patches from new image
    data0=X(:,s:s+T);   
    % different from training and test image                  
    test_data0 = X(:,s+T+100:s+2*T+100);
end
