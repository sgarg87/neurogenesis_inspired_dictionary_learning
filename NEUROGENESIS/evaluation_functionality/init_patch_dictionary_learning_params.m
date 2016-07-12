function params = init_patch_dictionary_learning_params(params, n, T)
    params.n = n;
    params.T = T;
    params.coding_sparse_algo = 'proximal';
    params.nonzero_frac = 0.10;
    params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
    params.dictionary_element_sparse_algo = 'proximal';
    params.nz_in_dict = 0.30; % number of nonzeros in each dictionary element
    params.new_elements = 0;  % new elements added per each batch of data
    params.batch_size = 20;  % batch size
    params.lambda_D = 0; % group sparsity
    params.epsilon = 1e-4; % convergence parameter for all methods
    params = rmfield(params, 'patch_size');
    params.is_init_A = true;
    params.is_init_B = false;
    params.A = 1e-3;
    params.B = 1e-10;
end
