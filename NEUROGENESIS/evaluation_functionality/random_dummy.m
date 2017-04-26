function [D, A, B, error, correlation] = random_dummy(train_data, D_init, params, A, B)
    % random-D: just use the D_init
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for random case.\n');
    params.lambda_D = 0;
    params.new_elements = -1;
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'Mairal', A, B);
end

