function [D, A, B, error, correlation] = neurogen_sg(train_data, D_init, params, A, B)
    % neurogenesis - with SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with SG.\n');
    [D,A, B, error,correlation] = DL(train_data, D_init, params, 'SG', A, B);
end

