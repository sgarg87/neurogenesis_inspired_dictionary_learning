function [data_train, data_test, num_pixels] = cifar_images_online(is_preprocess, num_data_per_label, dir_path)
    %
    if num_data_per_label ~= -1
        assert (num_data_per_label >= 1);
        assert (mod(500, num_data_per_label) == 0);
        assert (mod(100, num_data_per_label) == 0);
    end
    %     
    cifar_path = './Data/cifar-100-matlab/';
    cifar_path = strcat(dir_path, cifar_path);
    %     
    % training data    
    train = load(strcat(cifar_path, 'train.mat'));
    data_train = prepare_data_fr_raw(train, is_preprocess); clear train;
    num_pixels = size(data_train, 1);
    assert (size(data_train, 2) == 50000);
    %
    if num_data_per_label ~= -1
        data_train = data_train(:, 1:(500/num_data_per_label):end);
    end
    data_train = postprocess(data_train);
%     
% 
% 
% 
    % test data
    test = load(strcat(cifar_path, 'test.mat'));
    data_test = prepare_data_fr_raw(test, is_preprocess); clear test;
    assert (num_pixels == size(data_test, 1));
    assert (size(data_test, 2) == 10000);
    %     
    if num_data_per_label ~= -1 
        data_test = data_test(:, 1:(100/num_data_per_label):end);
    end
    %     
    data_test = postprocess(data_test);
end

function data_train = postprocess(data_train)
    num_data = size(data_train, 2);
    num_data_per_label = num_data/100;
    assert(mod(num_data_per_label, 1) == 0);
    %
    data_train_map = cell(100, 1);
    for curr_label = 1:100
        curr_idx = ((curr_label-1)*num_data_per_label)+1:(curr_label*num_data_per_label);
        data_train_map{curr_label} = data_train(:, curr_idx);
    end
    data_train = data_train_map; clear data_train_map;
end
    
function data_train = prepare_data_fr_raw(train, is_preprocess)
    data_train = train.data';
    data_train = convert_color_image_columns_to_gray_columns(data_train, 32, 32);
    % 
    [~, sort_idx] = sort(train.fine_labels);
    clear train;
    data_train = data_train(:, sort_idx);
    % preprocessing    
    if is_preprocess
        data_train = double(data_train)/255;
        data_train = preprocess_data(data_train);
    end
end
