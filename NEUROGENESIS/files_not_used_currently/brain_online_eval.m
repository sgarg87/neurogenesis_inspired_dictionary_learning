% Sahil Garg
% this function is an adaptation of the original function run_experiments.
function [new_elements,err0, correl0, learned_k0, err1, correl1, learned_k1, ...
    err2, correl2,learned_k2,err3, correl3, learned_k3, ...
    err4, correl4, learned_k4, err5, correl5, learned_k5] = brain_online_eval(params)
    % 
    k = params.k;
    % 
    fprintf('\n\n\n\n\n')
    fprintf('*******No. of dictionary elements to start is %d********.\n', k);
    % 
    eta = params.eta;
    epsilon = params.epsilon;
    D_update_method = params.D_update_method;
    new_elements = params.new_elements;
    lambda_D = params.lambda_D;
    mu = params.mu;
    data_type = params.data_type;
    nonzero_frac = params.nonzero_frac;
    %
    [train_data_map, test_data_map, n] = cifar_images_online(true);
    assert( params.n == n);
    % 
    D_init = normalize(rand(n,k));
    D0 = D_init; D1 = D_init; D2 = D_init; D3 = D_init; D4 = D_init; D5 = D_init;
    %
    labels = 1:100;
    %     
    for curr_label = labels
        %         
        train_data = train_data_map{curr_label};
        test_data = merge_data_map(test_data_map(1:curr_label), 1:curr_label);
        %         
        assert(size(train_data, 2) == params.T);
        T = params.T;
        %        
        % random-D: just use the D_init 
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for random case.\n');
        [D0,err00,correl00] = DL(train_data,D0,nonzero_frac,0,mu,eta,epsilon,T,-1,data_type,D_update_method);

        %neurogenesis - with GroupMairal 
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for neurogenesis with Group Mairal.\n');
        [D1,err11,correl11] =  DL(train_data,D1,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');

        % neurogenesis - with SG
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for neurogenesis with SG.\n');
        [D2,err22,correl22] = DL(train_data,D2,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');

        % group-sparse coding (Bengio et al 2009)
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for Group Mairal.\n');
        [D3,err33,correl33] = DL(train_data,D3,nonzero_frac,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal');
        % 
        % sahil: discuss this also with Dr. Rish.
        %%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly
        % SAHIL: lambda_D is zero in this case. so, this is not really sparse. (DISCUSS WITH DR. RISH.)
        %% fixed-size-SG
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for SG (no sparsity though).\n');
        [D4,err44,correl44] = DL(train_data,D4,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'SG');  
        % 
        % fixed-size-Mairal
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for Mairal.\n');
        [D5,err55,correl55] = DL(train_data,D5,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'Mairal');
        % 
        [~,err0,correl0] = sparse_coding(test_data,D0,nonzero_frac,data_type); % random-D
        [~,err1,correl1] = sparse_coding(test_data,D1,nonzero_frac,data_type);% neurogen-group-Mairal
        [~,err2,correl2] = sparse_coding(test_data,D2,nonzero_frac,data_type); % neurogen-SG
        [~,err3,correl3] = sparse_coding(test_data,D3,nonzero_frac,data_type);    % groupMairal 
        [~,err4,correl4] = sparse_coding(test_data,D4,nonzero_frac,data_type);    % SG       
        [~,err5,correl5] = sparse_coding(test_data,D5,nonzero_frac,data_type);    % Mairal    
        % 
        plot_online_err(params,err0,correl0,err1,correl1,err2,correl2,err3,correl3,err4,correl4,err5,correl5, strcat('nonstationarytest', num2str(curr_label)));
    end
    %
    clear train_data;
    clear test_data;
    %     
    learned_k0 = size(D0,2);
    learned_k1 = size(D1,2);
    learned_k2 = size(D2,2);
    learned_k3 = size(D3,2);
    learned_k4 = size(D4,2);
    learned_k5 = size(D5,2);
    %
    test_data = merge_data_map(test_data_map, labels);
    %
    [~,err0,correl0] = sparse_coding(test_data,D0,nonzero_frac,data_type); % random-D
    [~,err1,correl1] = sparse_coding(test_data,D1,nonzero_frac,data_type); % neurogen-group-Mairal
    [~,err2,correl2] = sparse_coding(test_data,D2,nonzero_frac,data_type); % neurogen-SG
    [~,err3,correl3] = sparse_coding(test_data,D3,nonzero_frac,data_type); % groupMairal 
    [~,err4,correl4] = sparse_coding(test_data,D4,nonzero_frac,data_type); % SG       
    [~,err5,correl5] = sparse_coding(test_data,D5,nonzero_frac,data_type); % Mairal    
end

function data = merge_data_map(data_map, labels)
    data = [];
    for curr_label = labels
        curr_data = data_map{curr_label};
        data = [data curr_data];
    end
end
