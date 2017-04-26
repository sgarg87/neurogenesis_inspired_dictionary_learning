function [D, A, B, error, correlation] = group_mairal(train_data, D_init, params, A, B)
    %%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly
    % group-sparse coding (Bengio et al 2009)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Group Mairal.\n');
    params.new_elements = 0;
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'GroupMairal', A, B);
end
