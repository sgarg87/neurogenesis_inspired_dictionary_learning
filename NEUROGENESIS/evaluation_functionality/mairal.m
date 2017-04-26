function [D, A, B, error, correlation] = mairal(train_data, D_init, params, A, B)
    % fixed-size-Mairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Mairal.\n');
    params.lambda_D = 0;
    params.new_elements = 0;
    [D,A, B, error,correlation] = DL(train_data, D_init, params, 'Mairal', A, B);
end

