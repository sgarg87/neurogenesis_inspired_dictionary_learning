function params = init_patch_dictionary_learning_params(params, n, T, nonzero_frac, is_sparse_dictionary, nz_in_dict, dict_size, epsilon)
    params.n = n;
    params.T = T;
    params.is_sparse_dictionary = is_sparse_dictionary; % sparse columns (elements) in dictionary
    params.nonzero_frac = nonzero_frac;
    params.nz_in_dict = nz_in_dict; % number of nonzeros in each dictionary element
    params.dict_size = dict_size;
    %     
    params.coding_sparse_algo = 'proximal';
    params.dictionary_element_sparse_algo = 'proximal';
    params.new_elements = 0;  % new elements added per each batch of data
    params.batch_size = 20;  % batch size
    params.lambda_D = 0; % group sparsity
    params.epsilon = epsilon; % convergence parameter for all methods
    params.is_init_A = true;
    params.is_init_B = false;
    params.A = 1e-3;
    params.B = 1e-10;
end
