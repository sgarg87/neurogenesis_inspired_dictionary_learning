function [D, A, B, error, correlation] = sg(train_data, D_init, params, A, B)
    % fixed-size-SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for SG (no sparsity though).\n');
    params.lambda_D = 0;
    params.new_elements = 0;
    [D, A, B, error,correlation] = DL(train_data,D_init, params, 'SG', A, B);
end
