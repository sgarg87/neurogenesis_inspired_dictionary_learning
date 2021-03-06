function model = initialize_model(dir_path)
    model = struct();
    %
    % setting seed zero for all random samplings.     
%     rng(0);
    %     
    model.params = init_parameters();
    model.algorithms = get_list_of_algorithms_fr_experiments();
    model.datasets_map = get_datasets_map(model.params.data_set_name, model.params.T, dir_path, model.params.n);
    %
    if model.params.is_patch_encoding
        if model.params.patch_multilayer
            model = learn_patch_layers(model);
        else
            patch_size = [10 10];
            nonzero_frac = 0.01;
            is_sparse_dictionary = true;
            nz_in_dict = 0.20;
            dict_size = 100;
            model.patches_data = get_patches(model.datasets_map, patch_size);
            model.patches_params = init_patch_dictionary_learning_params(model.params, size(model.patches_data.train, 1), size(model.patches_data.train, 2), nonzero_frac, is_sparse_dictionary, nz_in_dict, dict_size);
            model.patches_D = learn_patch_dictionary(model.patches_data.train, model.patches_params);
            model.patches_test_correlation = evaluate_patches_dictionary(model.patches_D, model.patches_data.test(:, 1:100:end), model.patches_params);
            model.datasets_map = get_patch_coding_fr_data(model.datasets_map, model.patches_D, model.patches_params, model.params.patch_size);
            %         
            model.params.n = size(model.datasets_map.data_nst_test, 1);
            save(strcat(dir_path, 'model_wd_patches'), 'model');
        end
    end
    %
    model.dictionary_sizes = get_dictionary_size_list_fr_algorithms(model.algorithms);
    %
    if model.algorithms.mairal
        curr_dictionary_sizes = model.dictionary_sizes.mairal;
        model.mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.random
        curr_dictionary_sizes = model.dictionary_sizes.random;
        model.random = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.group_mairal;
        model.group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.sg
        curr_dictionary_sizes = model.dictionary_sizes.sg;
        model.sg = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.neurogen_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_mairal;
        model.neurogen_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.neurogen_sg
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_sg;
        model.neurogen_sg = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.neurogen_group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_group_mairal;
        model.neurogen_group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
end


