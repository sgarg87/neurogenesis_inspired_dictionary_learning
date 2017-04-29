function [model, loss] = multi_armed_bandit(is_stationary, data_set_name, num_cycles, is_dictionary_coding, is_semi_supervised, is_weighted_rewards)
    % Y is supposed to be a vector, rather than a matrix.
    % However, we keep capital case for it, so as to have 
    % correspondingly x and y for a single data and the ground truth y.
    %     
    % 
    [X, Y] = get_data(data_set_name);
    %
    if is_dictionary_coding
        [X, dict_model] = get_dictionary_codes(X);
    end
    %     
    num_arms = length(unique(Y));
    %     
    if is_weighted_rewards
        p = hist(Y, unique(Y)); 
        p = power(p, -1); 
        p = p/sum(p);
    else
        p = ones(num_arms, 1);
    end
    % 
    num_features = size(X, 2);
    % 
    loss = zeros(num_cycles, 1);
    %
    %     
    for curr_idx =1:num_cycles
        if ((curr_idx == 1) || (~ is_stationary))
            model = initialize_arms_model(num_arms, num_features, is_semi_supervised, p);
        end
        %         
        [model, loss(curr_idx), correct_arms{curr_idx}, loss_arms{curr_idx}] = adapt_model_online(X, Y, model);
        fprintf('\n Mean Loss is %f.', mean(loss));
    end
    %
    %
    model.correct_arms = correct_arms;
    model.loss_arms = loss_arms;
    %     
    %     
    model.is_stationary = is_stationary;
    model.data_set_name = data_set_name;
    model.X = X;
    model.Y = Y;
    %     
    model.loss = loss;
    %     
    fprintf('\n Mean Loss is %f.\n', mean(loss));
    %
    if is_dictionary_coding
        model.dict_model = dict_model;
    end
    %
    if model.num_arms == 2
        precision = zeros(num_cycles, 1);
        recall = zeros(num_cycles, 1);
        f1_score = zeros(num_cycles, 1);
        %         
        for curr_idx =1:num_cycles
            precision(curr_idx) = model.correct_arms{curr_idx}(2) / (model.correct_arms{curr_idx}(2) + model.loss_arms{curr_idx}(1));
            recall(curr_idx) = model.correct_arms{curr_idx}(2) / (model.correct_arms{curr_idx}(2) + model.loss_arms{curr_idx}(2));
            f1_score(curr_idx) = (2*precision(curr_idx)*recall(curr_idx))/(precision(curr_idx)+recall(curr_idx));
        end
        %
        model.precision = precision;
        model.recall = recall;
        model.f1_score = f1_score;
        %         
        fprintf('\n Mean Precision is %f.', mean(model.precision));
        fprintf('\n Mean Recall is %f.', mean(model.recall));
        fprintf('\n Mean f1_score is %f.\n', mean(model.f1_score));
    end
    %     
    save model model;
end

function [C, dict_model] = get_dictionary_codes(X)
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
    curr_dict_size = 1500;
    curr_dictionary_sizes = [curr_dict_size];
    params = init_dict_parameters(num_dim, num_data);
    %
    % initialize model    
    dict_model = initialize_D_A_B(curr_dictionary_sizes, params);
    %     
    % 
    % learn the model, online on the data
    X_dict_lrn = datasample(X, params.T, 2, 'Replace', false);
    %     
    if is_mairal
        [dict_model.D{curr_dict_size}, ~, ~, ~, ~] = mairal(X_dict_lrn, dict_model.D{curr_dict_size}, params, dict_model.A{curr_dict_size}, dict_model.B{curr_dict_size});
    else
        [dict_model.D{curr_dict_size}, ~, ~, ~, ~] = neurogen_group_mairal(X_dict_lrn, dict_model.D{curr_dict_size}, params, dict_model.A{curr_dict_size}, dict_model.B{curr_dict_size});
    end
    %
    % removing zero elements from the dictionary    
    dict_model.D{curr_dict_size} = dict_model.D{curr_dict_size}(:, find(sum(dict_model.D{curr_dict_size}) ~= 0));
    %     
    dict_model.params = params;
    dict_model.dictionary_sizes = curr_dictionary_sizes;
    %     
    % evaluate model and also obtain the codes
    [C, error, correlation] = sparse_coding(X, dict_model.D{curr_dict_size}, params);
    %     
%     mean(error)
%     mean(correlation)
    %
    dict_model.error = error;
    dict_model.correlation = correlation;
    %     
    save dict_model dict_model;
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
    params.T = floor(num_data*0.1);  % total number of iterations/data samples
    params.coding_sparse_algo = 'proximal';
    params.nonzero_frac = 0.1;
%     params.nonzero_frac = 0.03;
    %     
    % proximal vs LARS
    params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
    params.dictionary_element_sparse_algo = 'proximal';
%     params.nz_in_dict = 0.0050; % number of nonzeros in each dictionary element
    params.nz_in_dict = 0.10; % number of nonzeros in each dictionary element
    % 
    params.lambda_D = 3e-2; % group sparsity
    %     
    params.is_sparse_data = true;
    %     
    params.new_elements = 50;  % new elements added per each batch of data
    %         
    params.batch_size = 200;  % batch size
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

function [model, loss, correct_arms, loss_arms] = adapt_model_online(X, Y, model)
    loss = 0;
    %     
    loss_arms = zeros(model.num_arms, 1);
    correct_arms = zeros(model.num_arms, 1);
    %     
    num_data = size(X, 1);
    assert (num_data == size(Y, 1));
    assert (size(Y, 2) == 1);
    %     
    count_inferences = 0;
    %
    %     
    % later on, process a batch of data, and include the module
    % for adaptation in a separate function
    for curr_data_idx = 1:num_data
        %         
        if model.is_semi_supervised
            if (mod(curr_data_idx, model.semisupervision_factor) ~= 1)
                continue;
            end
        end
        %         
        fprintf('\n ................%d..........................', curr_data_idx);
        %
        count_inferences = count_inferences + 1;
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
            try
                Sigma = model.Binv{curr_arm_idx}*power(model.v, 2);
                w{curr_arm_idx} = mvnrnd(mu, Sigma)';
            catch exception
                tic;
                model.Binv{curr_arm_idx} = nearestSPD(model.Binv{curr_arm_idx});
                fprintf('\n Time to nearest SPD: %f', toc);
                Sigma = model.Binv{curr_arm_idx}*power(model.v, 2);
                w{curr_arm_idx} = mvnrnd(mu, Sigma)';
            end
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
            correct_arms(y, 1) = correct_arms(y, 1) + 1;
        else
            reward = 0;
            loss = loss + model.p(y);
            loss_arms(y, 1) = loss_arms(y, 1) + 1;
        end
        %
        %         
        fprintf('\n reward: %d', reward);
        %         
        fprintf('\n loss ratio: %f', (loss/count_inferences));
        %         
        if reward == 1
            x_expr_fr_B_update = x'*x;
            model.B{max_reward_arm} = model.B{max_reward_arm} + x_expr_fr_B_update;
            clear x_expr_fr_B_update;
            tic;
            model.Binv{max_reward_arm} = inv(model.B{max_reward_arm});
            fprintf('\n time to inverse: %f', toc);
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
    loss = loss/count_inferences;
    % 
end


function model = initialize_arms_model(num_arms, context_size, is_semi_supervised, p)
    model.num_arms = num_arms;
    model.context_size = context_size;
    model.is_semi_supervised = is_semi_supervised;
    model.p = p;
    % 
    for curr_arm = 1:num_arms
        model.mu{curr_arm} = zeros(context_size, 1); 
        model.B{curr_arm} = eye(context_size);
        model.Binv{curr_arm} = inv(model.B{curr_arm});
        %         
        % thompson sampling parameter
        model.v = 0.25;
        model.f{curr_arm} = zeros(context_size, 1);
    end
    %    
%     model.semisupervision_factor = 20;
    model.semisupervision_factor = 100;
    % 
end

function [X, Y] = get_data(data_set_name)
    %
    if (data_set_name == 1)
        load cnae;
        X = cnae.data;
        Y = cnae.labels;
    elseif (data_set_name == 2)
        load covertype;
        X = covertype.data;
        Y = covertype.labels;
    elseif (data_set_name == 3)
        load semantic_paths_binary;
        X = semantic_paths_binary.data;
        Y = semantic_paths_binary.labels;
    elseif (data_set_name == 4)
        load semantic_path_kernel_hash_code;
        X = semantic_path_kernel_hash_code.data(:, 1:30); 
        Y = semantic_path_kernel_hash_code.labels;
    else
        assert false;
    end
    %     
end
