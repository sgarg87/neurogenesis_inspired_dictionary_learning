function [err0, correl0, learned_k0, err1, correl1, learned_k1] = analyze_group_sparsity_run_experiments(params)
    n = params.n;
    k = params.k;
    T = params.T;
    % 
    fprintf('\n\n\n\n\n')
    fprintf('************************No. of dictionary elements to start is %d******************************.\n', k);
    % 
    eta = params.eta;
    epsilon = params.epsilon;
    lambda_D = params.lambda_D;
    mu = params.mu;
    data_type = params.data_type;
    nonzero_frac = params.nonzero_frac;
    test_or_train = params.test_or_train;
    % 
    %%%%%%%%%%%%%%%%%%%%%%%% real images %%%%%%%%%%%%%%%%%%%%%%
    is_patches = false;
    % 
    if is_patches
         %real images (patches)
        [data0, test_data0] = boat_patches(T);
        [train_data, test_data, n] = lena_patches(T);
    else
        % or real images itself (Sahil)
        [train_data_map, test_data_map, n] = cifar_images_online(true, -1);
        % sea   
        train_data = train_data_map{72};
%         display(size(train_data));
        assert (size(train_data, 2) == T);
        test_data = test_data_map{72};
%         assert (size(test_data, 2) == T);
        % each column is a data point.
        data0 = train_data_map{89};
        assert (size(data0, 2) == T);
        test_data0 = test_data_map{89};
%         assert (size(test_data0, 2) == T);
    end
    % 
    D_init = normalize(rand(n,k));
    % 
    % group-sparse coding (Bengio et al 2009)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Group Mairal.\n');
    [D0,~,~] = DL(train_data,D_init,nonzero_frac,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal');

    % fixed-size-Mairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Mairal.\n');
    [D1,~,~] = DL(train_data,D_init,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'Mairal');
    % 
    assert(strcmp(test_or_train, 'nonstat'));
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for group Mairal.\n');
    [D0,~,~] = DL(data0,D0,nonzero_frac,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal'); %group Mairal
    %         
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Mairal.\n');
    [D1,~,~] = DL(data0,D1,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'Mairal');  %fixed-size Mairal
    % 
    [~,err0,correl0] = sparse_coding(test_data0,D0,nonzero_frac,data_type); % groupMairal
    [~,err1,correl1] = sparse_coding(test_data0,D1,nonzero_frac,data_type); % Mairal    
    % 
    learned_k0 = size(D0,2);
    learned_k1 = size(D1,2);
    % 
end