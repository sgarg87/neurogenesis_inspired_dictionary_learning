function [stationary_data_train, stationary_data_test, nonstationary_data_train, nonstationary_data_test, num_pixels] = cifar_images()
    cifar_path = './Data/cifar-100-matlab/';
%     
    train = load(strcat(cifar_path, 'train.mat'));    
    data_train = train.data';  
    num_pixels = size(data_train, 1);
    [~, sort_idx] = sort(train.fine_labels);
    clear train;
    stationary_data = data_train(:, sort_idx);
    clear data_train;
    % preprocessing    
    stationary_data = preprocess_data(stationary_data);
    %using this data_train to obtain both train and test for the model. stationary regime where some data samples from each class appear sequentially
    %originally there are 500 images in each class here.
    stationary_data_train = stationary_data(:, 1:25:end);
    stationary_data_test = stationary_data(:, 2:25:end);
    clear stationary_data;
% 
% 
    test = load(strcat(cifar_path, 'test.mat'));
    data_test = test.data';
    assert(size(data_test, 1) == num_pixels);
    clear test;
    % to induce nonstationarity into the data, simply shuffling the data. also, not doing the sorting contrary to the above.     
    num_test_data = size(data_test, 2);
    rand_permute_idx = randperm(num_test_data);    
    nonstationary_data = data_test(:, rand_permute_idx);
    clear data_test;
    % preprocessing
    nonstationary_data = preprocess_data(nonstationary_data);
    % equally splitting into train and test     
    nonstationary_data_train = nonstationary_data(:, 1:5:end);
    nonstationary_data_test = nonstationary_data(:, 2:5:end);
    clear nonstationary_data;
    