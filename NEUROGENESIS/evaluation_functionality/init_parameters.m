    function params = init_parameters()
    params = struct();
    %     
    params.n = 256;  % input size
    params.is_patch = true;  % patches vs images
    %
    params.eta = 0.1;  % parameter for SG
    params.adapt='basic'; %'adapt';
    params.epsilon = 1e-2; % convergence parameter for all methods
    params.mu = 0;   %  sparsity parameter for dictionary sparsity - now used only in SG version
    params.data_type = 'Gaussian';
    params.noise = 5;  % not used right now, only in simulated data
    %
    params.T = 100;  % total number of iterations/data samples
    params.new_elements = 10;  % new elements added per each batch of data
    params.batch_size = 20;  % batch size
    % group sparsity
    params.lambda_D = 0.03;
    %     
    % sparse codings
    params.nonzero_frac = 0.05;
    params.lambda2_C = 0.00001; %0;  % LASSO
    %
    % conditional neurogenesis related config.
    params.is_conditional_neurogenesis = false;
    params.errthresh = 0.5;
    %     
    % sparse columns (elements) in dictionary    
    params.is_sparse_dictionary = true;
    params.nz_in_dict = 0.05; % number of nonzeros in each dictionary element
    % params.True_nonzero_frac = 0.2;
    %
    params.is_reinitialize_dictionary_fixed_size = false;
    %     
    % initialization of the A, B matrices (prior brain memory)
    params.is_init_A = true;
    params.is_init_B = false;
    params.A = 1e-3;
    params.B = 1e-3;
    %
    % do not include the zero dictionary elements (columns in the
    % dictionary matrix) in final estimation of learned dictionary size
    % while plotting.
    params.is_nonzero_dict_element_in_learned_size = true;
    %
    params.is_sparse_computations = false;
end
