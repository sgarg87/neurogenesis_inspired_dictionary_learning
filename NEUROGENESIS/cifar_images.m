function [stationary_data_train, stationary_data_test, nonstationary_data_train, nonstationary_data_test, num_pixels] = cifar_images(is_preprocess)
    cifar_path = './Data/cifar-100-matlab/';
%     
    train = load(strcat(cifar_path, 'train.mat'));    
    data_train = train.data';
%     
    data_train = convert_color_image_columns_to_gray_columns(data_train, 32, 32);
% 
    num_pixels = size(data_train, 1);
    [~, sort_idx] = sort(train.fine_labels);
    clear train;
    stationary_data = data_train(:, sort_idx);
    clear data_train;
    % preprocessing    
    if is_preprocess     
        stationary_data = preprocess_data(stationary_data);
    end
    %using this data_train to obtain both train and test for the model. stationary regime where some data samples from each class appear sequentially
    %originally there are 500 images in each class here.
    stationary_data_train = stationary_data(:, 1:2:end);
    stationary_data_test = stationary_data(:, 2:2:end);
    clear stationary_data;
% 
% 
    test = load(strcat(cifar_path, 'test.mat'));
    data_test = test.data';
    clear test;
%     
    data_test = convert_color_image_columns_to_gray_columns(data_test, 32, 32);
% 
    assert(size(data_test, 1) == num_pixels);
    % to induce nonstationarity into the data, simply shuffling the data. also, not doing the sorting contrary to the above.     
    num_test_data = size(data_test, 2);
    rand_permute_idx = randperm(num_test_data);    
    nonstationary_data = data_test(:, rand_permute_idx);
    clear data_test;
    % preprocessing
    if is_preprocess
        nonstationary_data = preprocess_data(nonstationary_data);
    end
    % equally splitting into train and test     
    nonstationary_data_train = nonstationary_data(:, 1:2:end);
    nonstationary_data_test = nonstationary_data(:, 2:2:end);
    clear nonstationary_data;
    