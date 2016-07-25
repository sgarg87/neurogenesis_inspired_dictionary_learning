function params = init_parameters()
    params = struct();
    %
%     % 'patch', 'cifar', 'synthetic', 'nlp'
% 
%     params.data_set_name = 'cifar';  % patches vs images
%     params.n = 1024;  % input size
%     params.T = 100;  % total number of iterations/data samples
%     params.coding_sparse_algo = 'proximal';
%     params.nonzero_frac = 0.001;
%     params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
%     params.dictionary_element_sparse_algo = 'proximal';
%     params.nz_in_dict = 0.01; % number of nonzeros in each dictionary element
%     
% 
%     params.data_set_name = 'nlp';  % patches vs images
%     params.n = 12883;  % input size
%     params.T = 2750;  % total number of iterations/data samples
%     params.coding_sparse_algo = 'proximal';
%     params.nonzero_frac = 0.003;
%     % proximal vs LARS
%     params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
%     params.dictionary_element_sparse_algo = 'proximal';
%     params.nz_in_dict = 0.0020; % number of nonzeros in each dictionary element
% %     params.dict_element_lam = 1e0;
%
% 
% 
% 
    params.data_set_name = 'large_image';  % patches vs images
    params.n = 10000;  % input size
%     params.n = 65536;  % input size
    params.T = 1900;  % total number of iterations/data samples
    params.coding_sparse_algo = 'proximal';
    params.nonzero_frac = 0.01;
    % proximal vs LARS
    params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
    params.dictionary_element_sparse_algo = 'proximal';
    params.nz_in_dict = 0.01; % number of nonzeros in each dictionary element
%     params.dict_element_lam = 1e0;
% 
% 
% 
%     params.data_set_name = 'synthetic';  % patches vs images
%     params.n = 10000;  % input size
%     params.T = 1000;  % total number of iterations/data samples
%     params.coding_sparse_algo = 'proximal';
%     params.nonzero_frac = 0.05;
%     % proximal vs LARS
%     params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
%     params.dictionary_element_sparse_algo = 'proximal';
%     params.nz_in_dict = 0.05; % number of nonzeros in each dictionary element
% %     params.dict_element_lam = 1e-1;
    %
    %
    params.is_immunized_born_neurons = false;
    params.immunization_dose_fr_born_neurons = 1e-10;
    % in case of grand mother neurons, as of now, we add as many new dict. elements (neurons) as number of data.
    params.is_grand_mother_neurons = false;
    if params.is_grand_mother_neurons
        params.lambda_D = 0.0003; % group sparsity
    else
        params.lambda_D = 0.0003; % group sparsity
    end
    %
    params.new_elements = 1;  % new elements added per each batch of data
    %         
    params.batch_size = 20;  % batch size
%     params.batch_size = int64(params.T/5);  % batch size
    assert(params.batch_size <= params.T);
    %
    params.eta = 0.1;  % parameter for SG
    params.adapt='basic'; %'adapt';
    params.epsilon = 1e-2; % convergence parameter for all methods
    params.mu = 0;   %  sparsity parameter for dictionary sparsity - now used only in SG version
    params.data_type = 'Gaussian';
    params.noise = 5;  % not used right now, only in simulated data
    %     
    % sparse codings
    params.lambda2_C = 0.00001; %0;  % LASSO
    %
    % conditional neurogenesis related config.
    params.is_conditional_neurogenesis = false;
    params.errthresh = 0.5;
    %     
    % params.True_nonzero_frac = 0.2;
    %
    params.is_reinitialize_dictionary_fixed_size = false;
    %     
    % initialization of the A, B matrices (prior brain memory)
    params.is_init_A = false;
    params.is_init_B = false;
    params.A = 1e-10;
    params.B = 1e-3;
    params.is_A_sparse = false;
    params.A_sparse_nnz = 0.05;
    %
    % do not include the zero dictionary elements (columns in the
    % dictionary matrix) in final estimation of learned dictionary size
    % while plotting.
    params.is_nonzero_dict_element_in_learned_size = true;
    %
    params.is_sparse_computations = false;
    %
    params.is_sparse_dict_init = false;
    %
    params.is_patch_encoding = false;
    if params.is_patch_encoding
        params.patch_multilayer = true;
    end
end
