function [datasets_map] = get_datasets_map(data_set_name, T, dir_path, input_dim)
    if strcmp(data_set_name, 'patch')
        [train_data, test_data, data0, test_data0] = get_patches_data(T, dir_path, input_dim);
    elseif strcmp(data_set_name, 'cifar')
        [train_data, test_data, data0, test_data0] = get_cifar_data(T, dir_path, input_dim);
    elseif strcmp(data_set_name, 'synthetic')
        [train_data, test_data, data0, test_data0] = synthetic_sparse_data(input_dim, T, T);
    elseif strcmp(data_set_name, 'nlp')
        [train_data, test_data, data0, test_data0] = get_sparse_nlp_data();
    elseif strcmp(data_set_name, 'large_image')
        [train_data, test_data, data0, test_data0] = get_large_images(T, dir_path, input_dim);
    else
        error('no such data set');
    end
    %
    datasets_map = struct();
    datasets_map.data_st_train = train_data;
    datasets_map.data_st_test = test_data;
    datasets_map.data_nst_train = data0;
    datasets_map.data_nst_test = test_data0;
end


function [flowers_data_train, flowers_data_test, oxford_data_train, oxford_data_test] = get_large_images(T, dir_path, input_dim)
    n = sqrt(input_dim); clear input_dim;
    assert(mod(n, 1) == 0);
    image_size = [n n];
    %     
    [flowers_data, oxford_data] = flower_building_images_online(T*2, dir_path, image_size);
    %     
    flowers_data_train = flowers_data(:, 1:2:end);
    flowers_data_test = flowers_data(:, 2:2:end);
    %
    oxford_data_train = oxford_data(:, 1:2:end);
    oxford_data_test = oxford_data(:, 2:2:end);
end

function [train_data, test_data, data0, test_data0] = get_patches_data(T, dir_path, input_dim)
    %real images (patches)
    num_pixels_along_axis = round(sqrt(input_dim));
    patch_size = [num_pixels_along_axis num_pixels_along_axis];
    %     
    [data0, test_data0] = boat_patches(T, dir_path, patch_size);
    assert (size(data0, 1) == input_dim);
    assert (size(test_data0, 1) == input_dim);
    %     
    [train_data, test_data, ~] = lena_patches(T, dir_path, patch_size);
    assert (size(train_data, 1) == input_dim);
    assert (size(test_data, 1) == input_dim);
end

function [train_data, test_data, data0, test_data0] = get_cifar_data(T, dir_path, input_dim)
    [train_data_map, test_data_map, ~] = cifar_images_online(true, T, dir_path);    
    % sea images.
    st_idx = 14;
    nst_idx = 15;
    %     
    train_data = train_data_map{st_idx};
    test_data = test_data_map{st_idx};
    assert (size(train_data, 2) == T);
    assert (size(test_data, 2) == T);
    %     
    data0 = train_data_map{nst_idx};
    test_data0 = test_data_map{nst_idx};
    assert (size(data0, 2) == T);    
    assert (size(test_data0, 2) == T);
    %     
    assert(size(train_data, 1) == input_dim);
    assert(size(test_data, 1) == input_dim);
    assert(size(test_data0, 1) == input_dim);
    assert(size(data0, 1) == input_dim);
    %     
%     [train_data_map, ~, ~] = cifar_images_online(true, 100, dir_path);
%     data0 = [];
%     for curr_ns_label = 89:93
%         data0 = [data0 train_data_map{curr_ns_label}];
%     end
%     assert (size(data0, 2) == T);
%     %
%     assert(size(data0, 1) == input_dim);
end
