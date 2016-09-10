function [ datasets_map ] = get_patch_coding_fr_data(datasets_map, D, params_dict, patch_size)
    datasets_map.data_st_train = encode_dataset(datasets_map.data_st_train, D, params_dict, patch_size);
    datasets_map.data_nst_train = encode_dataset(datasets_map.data_nst_train, D, params_dict, patch_size);
    datasets_map.data_st_test = encode_dataset(datasets_map.data_st_test, D, params_dict, patch_size);
    datasets_map.data_nst_test = encode_dataset(datasets_map.data_nst_test, D, params_dict, patch_size);
end

function encoded_data = encode_dataset(data, D, params, patch_size)
    num_data = size(data, 2);
    % make the code more efficient by preallocation of memory
    encoding_size = size(get_data_encoding(data(:, 1), D, params, patch_size), 1);
    encoded_data = zeros(encoding_size, num_data);
    clear encoding_size;
    %     
    for curr_idx = 1:num_data
        curr_data = data(:, curr_idx);
        curr_data_encoding = get_data_encoding(curr_data, D, params, patch_size);
        encoded_data(:, curr_idx) = curr_data_encoding;
    end
end

function curr_data_sparse_coded = get_data_encoding(curr_data, D, params, patch_size)
    %% todo: move it to some outer function.
    patches_data = compute_patches(curr_data, patch_size);
    patches_codings = learn_encodings_fr_data_patches(patches_data, D, params);
    clear patches_data;
    %
    curr_data_sparse_coded = reshape(patches_codings, numel(patches_codings), 1);
    clear patches_codings;
end

function C = learn_encodings_fr_data_patches(curr_data_patches, D, params)
    [C,~,~] = sparse_coding(curr_data_patches, D, params);
end

