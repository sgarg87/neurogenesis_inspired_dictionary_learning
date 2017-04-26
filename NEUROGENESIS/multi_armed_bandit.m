function [model, loss] = multi_armed_bandit(is_stationary, data_set_name, num_cycles, is_dictionary_coding)
    % Y is supposed to be a vector, rather than a matrix.
    % However, we keep capital case for it, so as to have 
    % correspondingly x and y for a single data and the ground truth y.
    %     
    %     
    [X, Y] = get_data(data_set_name);
    %
    if is_dictionary_coding
        X = get_dictionary_codes(X);
    end
    %     
    num_arms = length(unique(Y));
    num_features = size(X, 2);
    % 
    loss = zeros(num_cycles, 1);
    %
    %     
    for curr_idx =1:num_cycles
        if ((curr_idx == 1) || (~ is_stationary))
            model = initialize_arms_model(num_arms, num_features);
        end
        %         
        [model, loss(curr_idx)] = adapt_model_online(X, Y, model);
        fprintf('\n Mean Loss is %f.', mean(loss));
    end
    %
    %     
    model.is_stationary = is_stationary;
    model.data_set_name = data_set_name;
    model.X = X;
    model.Y = Y;
    %     
    model.loss = loss;
    %     
    fprintf('\n Mean Loss is %f.', mean(loss));
    %     
    save model model;
end

function C = get_dictionary_codes(X)
    addpath('evaluation_functionality/');
    addpath 'ElasticNet/';
    %
    is_mairal = true;
    %     
    num_dim = size(X, 2);
    num_data = size(X, 1);
    %     
    X = X';
    %
    % initialize   
    curr_dict_size = 3000;
    curr_dictionary_sizes = [curr_dict_size];
    params = init_dict_parameters(num_dim, num_data);
    %
    % initialize model    
    dict_model = initialize_D_A_B(curr_dictionary_sizes, params);
    % learn the model, online on the data
    if is_mairal
        [dict_model.D{curr_dict_size}, ~, ~, ~, ~] = mairal(X, dict_model.D{curr_dict_size}, params, dict_model.A{curr_dict_size}, dict_model.B{curr_dict_size});
    else
        [dict_model.D{curr_dict_size}, ~, ~, ~, ~] = neurogen_group_mairal(X, dict_model.D{curr_dict_size}, params, dict_model.A{curr_dict_size}, dict_model.B{curr_dict_size});
    end
    %
    dict_model.params = params;
    dict_model.dictionary_sizes = curr_dictionary_sizes;
    save dict_model dict_model;
    %     
    % evaluate model and also obtain the codes
    [C, error, correlation] = sparse_coding(X, dict_model.D{curr_dict_size}, params);
    %
    dict_model.error = error;
    dict_model.correlation = correlation;
    % 
    C = C';
end

function params = init_dict_parameters(num_dim, num_data)
    params = init_parameters();
    %     
    params.data_set_name = 'multiarmedbandit';  % patches vs images
    %     
    params.n = num_dim;  % input size
    %     
    params.T = num_data;  % total number of iterations/data samples
    params.coding_sparse_algo = 'proximal';
    params.nonzero_frac = 0.1;
    %     
    % proximal vs LARS
    params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
    params.dictionary_element_sparse_algo = 'proximal';
    params.nz_in_dict = 0.0050; % number of nonzeros in each dictionary element
    % 
    params.lambda_D = 3e-2; % group sparsity
    %     
    params.is_sparse_data = true;
    %     
    params.new_elements = 50;  % new elements added per each batch of data
    %         
    params.batch_size = 50;  % batch size
    %     
    assert(params.batch_size <= params.T);
    %
    params.epsilon = 1e-2; % convergence parameter for all methods
    %     
    % conditional neurogenesis related config.
    params.is_conditional_neurogenesis = true;
    params.errthresh = 0.1;
    %     
    params.is_reinitialize_dictionary_fixed_size = false;
    %     
    % initialization of the A, B matrices (prior brain memory)
    params.is_init_A = false;
    params.is_init_B = false;
    params.is_A_sparse = false;
    %
    % do not include the zero dictionary elements (columns in the
    % dictionary matrix) in final estimation of learned dictionary size
    % while plotting.
    params.is_nonzero_dict_element_in_learned_size = true;
    %
    params.is_sparse_dict_init = false;
    %
    params.is_patch_encoding = false;
    %     
end

function [model, loss] = adapt_model_online(X, Y, model)
    loss = 0;
    %
    num_data = size(X, 1);
    assert (num_data == size(Y, 1));
    assert (size(Y, 2) == 1);
    %
    % later on, process a batch of data, and include the module 
    % for adaptation in a separate function
    for curr_data_idx = 1:num_data
        fprintf('\n ................%d..........................', curr_data_idx);
        %         
        x = X(curr_data_idx, :);
        y = Y(curr_data_idx, 1);
        %
        reward_inference = zeros(model.num_arms, 1);
        %         
        for curr_arm_idx =1:model.num_arms
            % sample the weights from the normal distribution
            mu = model.mu{curr_arm_idx};
            %             
            Sigma = model.Binv{curr_arm_idx}*power(model.v, 2);
%             Sigma = Sigma + diag(diag(Sigma))*0.3;
            w{curr_arm_idx} = mvnrnd(mu, Sigma)';
            %             
            reward_inference(curr_arm_idx, 1) = x*w{curr_arm_idx};
        end
        %
        clear w;
        clear curr_arm_idx;
        %         
        [max_reward_inference, max_reward_arm] = max(reward_inference);
        %
        fprintf('\n max_reward_inference: %f', max_reward_inference);
        fprintf('\n max_reward_arm: %d', max_reward_arm);
        %         
        clear max_reward_inference;
        %
        % play the arm
        if (max_reward_arm == y)
            reward = 1;
        else
            reward = 0;
        end
        %  
        fprintf('\n reward: %d', reward);
        %         
        loss = loss + (1-reward);
        %         
        fprintf('\n loss: %d', loss);
        %         
        if reward == 1
            x_expr_fr_B_update = x'*x;
            model.B{max_reward_arm} = model.B{max_reward_arm} + x_expr_fr_B_update;
            clear x_expr_fr_B_update;
            model.Binv{max_reward_arm} = inv(model.B{max_reward_arm});
            model.Binv{max_reward_arm} = nearestSPD(model.Binv{max_reward_arm});
            %         
            x_expr_fr_f_update = x'*reward;
            model.f{max_reward_arm} = model.f{max_reward_arm} + x_expr_fr_f_update;
            clear x_expr_fr_f_update;
            %         
            model.mu{max_reward_arm} = model.Binv{max_reward_arm}*model.f{max_reward_arm};
            %         
        end
    end
    %
    loss = loss/num_data;
    % 
end


function model = initialize_arms_model(num_arms, context_size)
    model.num_arms = num_arms;
    model.context_size = context_size;
    % 
    for curr_arm = 1:num_arms
        model.mu{curr_arm} = zeros(context_size, 1); 
        model.B{curr_arm} = eye(context_size);
        model.Binv{curr_arm} = inv(model.B{curr_arm});
        %         
        % thompson sampling parameter
        model.v = 0.2;
        model.f{curr_arm} = zeros(context_size, 1);
    end
end

function [X, Y] = get_data(data_set_name)
    %
    if strcmp(data_set_name, 'cnae')
        load cnae;
        X = cnae.data;
        Y = cnae.labels;
    elseif strcmp(data_set_name, 'sp')
        load semantic_paths_binary;
        X = semantic_paths_binary.data;
        Y = semantic_paths_binary.labels;
    else
        assert false;
    end
    %     
end

