function [D, A, B, error, correlation] =  neurogen_group_mairal(train_data, D_init, params, A, B)
    %neurogenesis - with GroupMairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Group Mairal.\n');
    % 'GroupMairal'
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'GroupMairal', A, B);
end

