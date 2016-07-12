function D = learn_patch_dictionary(train_data, params)
    curr_dict_size = 1000;
    %
    n = size(train_data, 1);
    assert(n == params.n);
    T = size(train_data, 2);
    assert(T == params.T);
    %     
    D_init = normalize(rand(n,curr_dict_size));
    %     
    % 
    if params.is_init_A
        curr_random_init = rand(curr_dict_size, curr_dict_size);
        A = params.A*(curr_random_init'*curr_random_init);
    else
        A = [];
    end
    %
    if params.is_init_B
        B = params.B*rand(n, curr_dict_size);
    else
        B = [];
    end
    %     
    %     
    fprintf('Learning the patch dictionary using standard Mairal online method.\n');
    [D,~, ~, ~,~] = DL(train_data, D_init, params, 'Mairal', A, B);    
end
