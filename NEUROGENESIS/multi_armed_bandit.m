function [model, loss] = multi_armed_bandit(is_stationary, data_set_name, num_cycles, is_dictionary_coding, is_semi_supervised, is_weighted_rewards)
    %
    %     
    % Y is supposed to be a vector, rather than a matrix.
    % However, we keep capital case for it, so as to have 
    % correspondingly x and y for a single data and the ground truth y.
    %     
    % 
    [X, Y] = get_data(data_set_name);
    %
    if is_dictionary_coding
        dict_model = get_dictionary_codes(X, data_set_name);
    else
        dict_model = [];
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
    if is_dictionary_coding
        num_features = size(dict_model.D, 2);
    else
        num_features = size(X, 2);
    end
    % 
    loss = zeros(num_cycles, 1);
    %
    %     
    for curr_idx =1:num_cycles
        if ((curr_idx == 1) || (~ is_stationary))
            model = initialize_arms_model(num_arms, num_features, is_semi_supervised, p, data_set_name);
            %
            model.is_weighted_rewards = is_weighted_rewards;
            %             
            model.is_dictionary_coding = is_dictionary_coding;
            %             
            if is_dictionary_coding
                model.dict_model = dict_model;
            end
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
    fprintf('\n std Loss is %f.\n', std(loss));
    %
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

function [dict_model] = get_dictionary_codes(X, data_set_name)
    %     
    addpath('evaluation_functionality/');
    addpath 'ElasticNet/';
    %
    %     
    num_dim = size(X, 2);
    num_data = size(X, 1);
    %     
    X = X';
    %
    % initialize
    %     
    if data_set_name == 1 %CNAE
        curr_dict_size = 3000;
    elseif data_set_name == 2 %Covertype
        curr_dict_size = 300;
    elseif data_set_name == 6 %Poker
        curr_dict_size = 100;
    elseif data_set_name == 5 %Internet Ad click
        curr_dict_size = 3000;
    elseif data_set_name == 4 %Kernel hash codes
        curr_dict_size = 100;
    elseif data_set_name == 8 %FAO
        curr_dict_size = 2000;
    else
        assert false;
    end
    % 
    %     
    curr_dictionary_sizes = [curr_dict_size];
    params = init_dict_parameters(num_dim, num_data, data_set_name);
    %
    % initialize model    
    dict_model = initialize_D_A_B(curr_dictionary_sizes, params);
    %     
    dict_model.D = dict_model.D{curr_dict_size};
    dict_model.A = 1e-10*ones(curr_dict_size, curr_dict_size);
%     dict_model.A = dict_model.A{curr_dict_size};
    dict_model.B = 1e-10*ones(size(dict_model.D));
%     dict_model.B = dict_model.B{curr_dict_size};
    % 
    if params.T ~= 0
        % learn the model, online on the data
        X_dict_lrn = datasample(X, params.T, 2, 'Replace', false);
        %     
        if params.is_mairal
            [dict_model.D, dict_model.A, dict_model.B, ~, ~] = mairal(X_dict_lrn, dict_model.D, params, dict_model.A, dict_model.B);
        else
            [dict_model.D, dict_model.A, dict_model.B, ~, ~] = neurogen_group_mairal(X_dict_lrn, dict_model.D, params, dict_model.A, dict_model.B);
        end
        %
%         random initialize zero elements
%         zero_idx = find(sum(dict_model.D) == 0);
%         dict_model.D(:, zero_idx) = normalize(rand(size(dict_model.D, 1), length(zero_idx)));
%         clear zero_idx;
        %         
%         % removing zero elements from the dictionary
        nonzero_idx = find(sum(dict_model.D) ~= 0);
        dict_model.D = dict_model.D(:, nonzero_idx);
        dict_model.A = dict_model.A(nonzero_idx, nonzero_idx);
        dict_model.B = dict_model.B(:, nonzero_idx);
        clear nonzero_idx;
    end
    %     
    dict_model.params = params;
    dict_model.dictionary_sizes = curr_dictionary_sizes;
    %
%     % evaluate model and also obtain the codes
%     [C, error, correlation] = sparse_coding(X, dict_model.D{curr_dict_size}, params);
    %
%     dict_model.error = error;
%     dict_model.correlation = correlation;
    %     
    save dict_model dict_model;
    % 
%     C = C';
end

function params = init_dict_parameters(num_dim, num_data, data_set_name)
    %
    %     
    params = init_parameters();
    %     
    params.is_mairal = true;
    %     
    params.data_set_name = 'multiarmedbandit';  % patches vs images
    %     
    params.n = num_dim;  % input size
    %
    params.T = max(200, floor(num_data*0.01));  % total number of iterations/data samples
    params.coding_sparse_algo = 'proximal';
    %     
    if data_set_name == 1 %CNAE
        params.nonzero_frac = 0.1;
    elseif data_set_name == 2 %Covertype
        params.nonzero_frac = 0.1;
    elseif data_set_name == 6 %Poker
        params.nonzero_frac = 1;
    elseif data_set_name == 5 %Internet Ad click
        params.nonzero_frac = 0.03;
    elseif data_set_name == 4 %Kernel hash codes
        params.nonzero_frac = 2.5;
    elseif data_set_name == 8 %FAO
        params.nonzero_frac = 0.2;
    else
        assert false;
    end
    % 
    %     
    % proximal vs LARS
    params.is_sparse_dictionary = true; % sparse columns (elements) in dictionary
    params.dictionary_element_sparse_algo = 'proximal';
    %     
    %     
    if data_set_name == 1 %CNAE
        params.nz_in_dict = 0.005;
    elseif data_set_name == 2 %Covertype
        params.nz_in_dict = 0.1;
    elseif data_set_name == 6 %Poker
        params.nz_in_dict = 0.3;
    elseif data_set_name == 5 %Internet Ad click
        params.nz_in_dict = 0.005; % number of nonzeros in each dictionary element
    elseif data_set_name == 4 %Kernel hash codes
        params.nz_in_dict = 0.1; % number of nonzeros in each dictionary element
    elseif data_set_name == 8 %FAO
        params.nz_in_dict = 0.005; % number of nonzeros in each dictionary element
    else
        assert false;
    end
    %     
    % 
    params.lambda_D = 3e-2; % group sparsity
    %     
    %     
    if data_set_name == 1 %CNAE
        params.is_sparse_data = true;
    elseif data_set_name == 2 %Covertype
        params.is_sparse_data = true;
    elseif data_set_name == 6 %Poker
        params.is_sparse_data = true;
    elseif data_set_name == 5 %Internet Ad click
        params.is_sparse_data = true;
    elseif data_set_name == 4 %Kernel hash codes
        params.is_sparse_data = true;
    elseif data_set_name == 8 %FAO
        params.is_sparse_data = false;
    else
        assert false;
    end
    %
    %     
    params.new_elements = 50;  % new elements added per each batch of data
    %         
    params.batch_size = min(params.T, 200);  % batch size
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
    %
    %     
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
    %     
    for curr_data_idx = 1:num_data
        %         
        %
        if model.is_dictionary_coding
            if (mod(curr_data_idx, model.online_dict_update_batch_size) == 0)
                fprintf('\n *********************%d***********************', curr_data_idx);
                %                 
                X_dict_lrn = X(curr_data_idx-model.online_dict_update_batch_size+1:curr_data_idx, :)';
                %
                model.dict_model.params.batch_size = model.online_dict_update_batch_size;
                model.dict_model.params.T = model.online_dict_update_batch_size;
                model.dict_model.params.is_reinitialize_dictionary_fixed_size = false;
                % 
                %
                if model.dict_model.params.is_mairal
                    [model.dict_model.D, model.dict_model.A, model.dict_model.B, ~, ~] = mairal(X_dict_lrn, model.dict_model.D, model.dict_model.params, model.dict_model.A, model.dict_model.B);
                else
                    [model.dict_model.D, model.dict_model.A, model.dict_model.B, ~, ~] = neurogen_group_mairal(X_dict_lrn, model.dict_model.D, model.dict_model.params, model.dict_model.A, model.dict_model.B);
                end
                %                 
                clear X_dict_lrn;
            end
        end
        %         
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
        if model.is_dictionary_coding
            [c, error, correlation] = sparse_coding(x', model.dict_model.D, model.dict_model.params);
            x = c';
            error
            correlation
        end
        %         
        reward_inference = zeros(model.num_arms, 1);
        %         
        for curr_arm_idx =1:model.num_arms
            % sample the weights from the normal distribution
            mu = model.mu{curr_arm_idx};
            %
            if model.is_approximate_sampling
%                 old code
%                     tic;
%                     w_rnd = mvnrnd(model.mu_zero, model.B_eye)';
%                     w{curr_arm_idx} = mu + lsqr(model.Bchol{curr_arm_idx}', w_rnd)*model.v;
%                     fprintf('\n Time to approximate sampling: %f.\n', toc);
                    %            
                    % new code        
                    tic;
%                     w_rnd = mvnrnd(model.mu_zero, model.B_eye)';
                    w_rnd = randn(model.context_size, 1);
%                     fprintf('\n Time to random sampling: %f.\n', toc);
%                     tic;
                    w{curr_arm_idx} = mu + (model.Bcholinv{curr_arm_idx}')*w_rnd*model.v;
%                     fprintf('\n Time to project the sampling: %f.\n', toc);
                    fprintf('\n Time to sampling: %f.\n', toc);
            else
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
            % 
            if model.num_arms == 2
                max_reward_arm = y;
            end
        end
        %
        %         
        fprintf('\n reward: %d', reward);
        %         
        fprintf('\n loss ratio: %f', (loss/count_inferences));
        %         
        if (max_reward_arm == y)
            x_expr_fr_B_update = x'*x;
            model.B{max_reward_arm} = model.B{max_reward_arm} + x_expr_fr_B_update;
            clear x_expr_fr_B_update;
            %
            if model.is_approximate_sampling
                tic;
                model.Bchol{max_reward_arm} = cholupdate(model.Bchol{max_reward_arm}, x', '+');
                fprintf('\n time to update cholsky factorization: %f. \n', toc);
                tic;
                model.Bcholinv{max_reward_arm} = inv(model.Bchol{max_reward_arm});
                fprintf('\n time to chol inverse: %f', toc);
                tic;
                model.Binv{max_reward_arm} = (model.Bcholinv{max_reward_arm})*(model.Bcholinv{max_reward_arm})';
                fprintf('\n time to inverse: %f', toc);
            else
                tic;
                model.Binv{max_reward_arm} = inv(model.B{max_reward_arm});
                fprintf('\n time to inverse: %f', toc);
            end
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


function model = initialize_arms_model(num_arms, context_size, is_semi_supervised, p, data_set_name)
    %
    %     
    model.num_arms = num_arms;
    model.context_size = context_size;
    model.is_semi_supervised = is_semi_supervised;
    model.p = p;
    % 
    model.is_approximate_sampling = true;
    %     
    model.mu_zero = zeros(context_size, 1); 
    model.B_eye = eye(context_size);
    %         
    for curr_arm = 1:num_arms
        model.mu{curr_arm} = zeros(context_size, 1); 
        model.B{curr_arm} = eye(context_size);
        %         
        if model.is_approximate_sampling
            model.Bchol{curr_arm} = chol(model.B{curr_arm});
            model.Bcholinv{curr_arm} = inv(model.Bchol{curr_arm});
        else
            model.Binv{curr_arm} = inv(model.B{curr_arm});
        end
        %         
        % thompson sampling parameter
        model.v = 0.25;
        model.f{curr_arm} = zeros(context_size, 1);
    end
    %
    if data_set_name == 1 %CNAE
        model.semisupervision_factor = 2;
    elseif data_set_name == 2 %Covertype
        model.semisupervision_factor = 100;
    elseif data_set_name == 6 %Poker
        model.semisupervision_factor = 20;
    elseif data_set_name == 5 %Internet Ad click
        model.semisupervision_factor = 5;
    elseif data_set_name == 4 %Kernel hash codes
        model.semisupervision_factor = 20;
    elseif data_set_name == 8 %FAO
        model.semisupervision_factor = 20;
    else
        assert false;
    end
    % 
    model.online_dict_update_batch_size = 200;
end

function [X, Y] = get_data(data_set_name)
    %
    %     
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
    elseif (data_set_name == 5)
        load internetad;
        X = internetad.data; 
        Y = internetad.labels;
    elseif (data_set_name == 6)
        load pokerhand;
        X = pokerhand.data; 
        Y = pokerhand.labels;
    elseif (data_set_name == 7)
        load cifar;
        X = cifar.data; 
        Y = cifar.labels;
    elseif (data_set_name == 8)
        load fao;
        X = fao.data; 
        Y = fao.labels;
    else
        assert false;
    end
    %     
end

