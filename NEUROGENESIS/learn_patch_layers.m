function model = learn_patch_layers(model)
    curr_layer = learn_layer(model.datasets_map, model.params, [8 8], 0.4, true, 0.05, 256, 1e-2);
%     
%     curr_layer = learn_layer(model.datasets_map, model.params, [4 4], 12.5, true, 0.80, 1024, 1e-4);
    %     
%     curr_layer = learn_layer(curr_layer.datasets_map, model.params, [8 8], 0.4, true, 0.8, 64, 1e-3);
    %     
%     curr_layer = learn_layer(curr_layer.datasets_map, model.params, [16 16], 0.1, true, 0.2, 256, 1e-3);
    %     
%     curr_layer = learn_layer(curr_layer.datasets_map, model.params, [32 32], 0.1, true, 0.2, 1024, 1e-3);
    %     
%     curr_layer = learn_layer(curr_layer.datasets_map, model.params, [64 64], 0.01, true, 0.1, 4096, 1e-2);
    %     
    model.datasets_map = curr_layer.datasets_map;
    clear curr_layer;    
    model.params.n = size(model.datasets_map.data_nst_test, 1);
    save model model;
end

function model = learn_layer(datasets_map, params, patch_size, nonzero_frac, is_sparse_dictionary, nz_in_dict, dict_size, epsilon)
    model = struct();
    model.patches_data = get_patches(datasets_map, patch_size);
    %
    %     
    % we need more larger subsection from large data (small patches)
    % todo: there may be a better way to do it. it seems naive as it is now.
    train_data = model.patches_data.train;
%     if size(train_data, 2)/1e4 > 1e4
%         train_data = train_data(:, 1:1e4:end);
%     elseif size(train_data, 2)/1e3 > 1e4
%         train_data = train_data(:, 1:1e3:end);
%     elseif size(train_data, 2)/100 > 1e4
%         train_data = train_data(:, 1:100:end);
%     elseif size(train_data, 2)/10 > 1e4
%         train_data = train_data(:, 1:10:end);
%     end
    model.patches_params = init_patch_dictionary_learning_params(params, size(train_data, 1), size(train_data, 2), nonzero_frac, is_sparse_dictionary, nz_in_dict, dict_size, epsilon);
    model.patches_D = learn_patch_dictionary(train_data, model.patches_params);
    %
    %     
    test_data = model.patches_data.test;
    if size(test_data, 2)/100 > 1e4
        test_data = test_data(:, 1:100:end);
    elseif size(test_data, 2)/10 > 1e4
        test_data = test_data(:, 1:10:end);
    end        
    model.patches_test_correlation = evaluate_patches_dictionary(model.patches_D, test_data, model.patches_params);
    display(mean(model.patches_test_correlation, 2));
    %     
    % 
    model.datasets_map = get_patch_coding_fr_data(datasets_map, model.patches_D, model.patches_params, patch_size);
end

