function [datasets_map] = get_datasets_map(data_set_name, T, dir_path, input_dim)
    if strcmp(data_set_name, 'patch')
        [train_data, test_data, data0, test_data0] = get_patches_data(T, dir_path, input_dim);
    elseif strcmp(data_set_name, 'cifar')
        [train_data, test_data, data0, test_data0] = get_cifar_data(T, dir_path, input_dim);
    elseif strcmp(data_set_name, 'synthetic')
        [train_data, test_data, data0, test_data0] = synthetic_sparse_data(input_dim, T, T);
    elseif strcmp(data_set_name, 'nlp')
        [train_data, test_data, data0, test_data0] = get_sparse_nlp_data();
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
