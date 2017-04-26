function [D, A, B, error, correlation] = neurogen_mairal(train_data, D_init, params, A, B)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Mairal.\n');
    params.lambda_D = 0;
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'Mairal', A, B);
end

