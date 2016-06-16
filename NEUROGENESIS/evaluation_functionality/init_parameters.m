function params = init_parameters()
    params = struct();
    %     
    params.n = 256;
    params.is_patch = true;
    %     
    params.eta = 0.1;
    params.adapt='basic'; %'adapt';
    params.epsilon = 1e-2;
    params.mu = 0;
    params.data_type = 'Gaussian';
    params.noise = 5;
    %     
    params.T = 100;
    params.new_elements = 100;
    params.batch_size = 20;
    % group sparsity    
    params.lambda_D = 0.03;
    % sparse codings    
    params.nonzero_frac = 0.05;
    %     
    params.is_conditional_neurogenesis = false;
    % sparse columns (elements) in dictionary    
    params.is_sparse_dictionary = true;
    params.nz_in_dict = 0.01;
%     
%     params.True_nonzero_frac = 0.2;
end
