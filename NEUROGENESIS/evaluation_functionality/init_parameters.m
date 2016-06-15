function params = init_parameters()
    % todo: remove parameters that are not being used.
    params = struct();
    params.n = 1024;
    params.eta = 0.1;
    params.adapt='basic'; %'adapt';
    params.T = 100;
    params.new_elements = 40;
    params.epsilon = 1e-2;
    params.lambda_D = 0.03;
    params.mu = 0;
    params.data_type = 'Gaussian';
    params.noise = 5;
    params.True_nonzero_frac = 0.2;
    params.nonzero_frac = 0.05;
    params.is_patch = true;
end
