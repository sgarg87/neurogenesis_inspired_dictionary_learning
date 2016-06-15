% This script is written by Sahil Garg, consolidating the functionality of
% the original scripts main_online and run_experiments.
% This is more flexible in terms of choosing which algorithms to evaluate
% in the simulations (experiments).
% 
function evaluate_online(is_hpcc)
    dir_path = get_dir_path(is_hpcc);
    % 
    model = initialize_model(dir_path);
    % 
    model = adapt_model(model, model.datasets_map.data_st_train);
    model = adapt_model(model, model.datasets_map.data_nst_train);
    %     
    model.evaluation = evaluate_model(model);
    % 
    plot_correlation(model, dir_path);
    plot_learned_dictionary_size(model, dir_path);
    %
    save(strcat(dir_path, 'model'), 'model');
end

function params = init_parameters()
    params = struct();
    params.n = 256;
    params.eta = 0.1;
    params.adapt='basic'; %'adapt';
    params.T = 100;
    params.new_elements = 10;
    params.epsilon = 1e-2;
    params.lambda_D = 0.03;
    params.mu = 0;
    params.data_type = 'Gaussian';
    params.noise = 5;
    params.True_nonzero_frac = 0.2;
    params.nonzero_frac = 0.20;
    params.is_patch = true;
end

function dir_path = get_dir_path(is_hpcc)
	if is_hpcc
		% todo: correct this path, not complete.
		dir_path = '/auto/rcf-proj2/gv/sahilgar/sparse_dictionary_learning/code/neurogenesis_irina_rish/NEUROGENESIS/';
	else
		dir_path = './';
	end
end

function algorithms = get_list_of_algorithms_fr_experiments()
    algorithms = struct();
    %     
    algorithms.mairal = false;
    algorithms.random = false;
    algorithms.group_mairal = false;
    algorithms.neurogen_group_mairal = false;
    algorithms.neurogen_mairal = false;
    algorithms.neurogen_sg = false;
    algorithms.sg = false;
    %     
    % setting it to false first and then true so that we can simply comment
    % the algorithms we don't want to use while running the experiments.     
    %     
    algorithms.mairal = true;
    algorithms.neurogen_mairal = true;
    %     
    algorithms.random = true;
    algorithms.group_mairal = true;
    algorithms.neurogen_group_mairal = true;
    algorithms.neurogen_sg = true;
    algorithms.sg = true;
end

function obj = initialize_D_A_B(curr_dictionary_sizes, n)
    D_init = {};
    A = {};
    B = {};
    for curr_k = curr_dictionary_sizes
        D_init{curr_k} = normalize(rand(n,curr_k));            
        A{curr_k} = [];
        B{curr_k} = [];
    end
    obj = struct();
    obj.D = D_init;
    obj.A = A;
    obj.B = B;
end

function model = initialize_model(dir_path)
    model = struct();
    % 
    model.params = init_parameters();
    model.algorithms = get_list_of_algorithms_fr_experiments();
    model.datasets_map = get_datasets_map(model.params.is_patch, model.params.T, dir_path);
    %
    model.dictionary_sizes = get_dictionary_size_list_fr_algorithms(model.algorithms);
    rng(0);
    %     
    if model.algorithms.mairal
        curr_dictionary_sizes = model.dictionary_sizes.mairal;
        model.mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.random
        curr_dictionary_sizes = model.dictionary_sizes.random;
        model.random = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.group_mairal;
        model.group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.sg
        curr_dictionary_sizes = model.dictionary_sizes.sg;
        model.sg = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.neurogen_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_mairal;
        model.neurogen_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.neurogen_sg
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_sg;
        model.neurogen_sg = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.neurogen_group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_group_mairal;
        model.neurogen_group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
end

function [dictionary_sizes] = get_dictionary_size_list_fr_algorithms(algorithms)
    dictionary_sizes = struct();
    %
    global_size = [25 50 100 150];
    %     
    if algorithms.mairal
        dictionary_sizes.mairal = [25 50 100 150];
    end
    %
    if algorithms.random
        dictionary_sizes.random = [25 50 100 150];
    end
    %
    if algorithms.group_mairal
        dictionary_sizes.group_mairal = [25 50 100 150];
    end
    %
    if algorithms.sg
        dictionary_sizes.sg = [25 50 100 150];
    end
    %
    if algorithms.neurogen_mairal
        dictionary_sizes.neurogen_mairal = [25 50 100 150];
    end
    %
    if algorithms.neurogen_sg
        dictionary_sizes.neurogen_sg = [25 50 100 150];
    end
    %
    if algorithms.neurogen_group_mairal
        dictionary_sizes.neurogen_group_mairal = [25 50 100 150];
    end 
end

function curr_eval_obj = create_evaluation_object(error, correlation)
    curr_eval_obj = struct();
    curr_eval_obj.error = error;
    curr_eval_obj.correlation = correlation;
end

function evaluation = evaluate_model(model)
    evaluation = struct();
    algorithms = model.algorithms;
    params = model.params;
    test_data = model.datasets_map.data_nst_test;
    dictionary_sizes = model.dictionary_sizes;
    %     
    if algorithms.random
        evaluation.random = {};
        % 
        for curr_dict_size = dictionary_sizes.random
            D = model.random.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.random{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.mairal
        evaluation.mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.mairal
            D = model.mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.group_mairal
        evaluation.group_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.group_mairal
            D = model.group_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.group_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.sg
        evaluation.sg = {};
        % 
        for curr_dict_size = dictionary_sizes.sg
            D = model.sg.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.sg{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.neurogen_group_mairal
        evaluation.neurogen_group_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_group_mairal
            D = model.neurogen_group_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.neurogen_group_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.neurogen_sg
        evaluation.neurogen_sg = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_sg
            D = model.neurogen_sg.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.neurogen_sg{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.neurogen_mairal
        evaluation.neurogen_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_mairal
            D = model.neurogen_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.neurogen_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
end

function model = adapt_model(model, train_data)
    algorithms = model.algorithms;
    dictionary_sizes = model.dictionary_sizes;
    params = model.params;
    %
    if algorithms.random
        for curr_dict_size = dictionary_sizes.random
            D_init = model.random.D{curr_dict_size};
            A = model.random.A{curr_dict_size};
            B = model.random.B{curr_dict_size};
            [D,A,B, ~,~] = random_dummy(train_data, D_init, params, A, B);
            model.random.D{curr_dict_size} = D;
            model.random.A{curr_dict_size} = A;
            model.random.B{curr_dict_size} = B;
        end
    end
    %     
    if algorithms.mairal
        for curr_dict_size = dictionary_sizes.mairal
            D_init = model.mairal.D{curr_dict_size};
            A = model.mairal.A{curr_dict_size};
            B = model.mairal.B{curr_dict_size};
            [D,A,B,~,~] = mairal(train_data, D_init, params, A, B);
            model.mairal.D{curr_dict_size} = D;
            model.mairal.A{curr_dict_size} = A;
            model.mairal.B{curr_dict_size} = B;
        end
    end
    %
    if algorithms.group_mairal
        for curr_dict_size = dictionary_sizes.group_mairal
            D_init = model.group_mairal.D{curr_dict_size};
            A = model.group_mairal.A{curr_dict_size};
            B = model.group_mairal.B{curr_dict_size};
            [D,A,B,~,~] = group_mairal(train_data, D_init, params, A, B);
            model.group_mairal.D{curr_dict_size} = D;
            model.group_mairal.A{curr_dict_size} = A;
            model.group_mairal.B{curr_dict_size} = B;
        end
    end
    %     
    if algorithms.sg
        for curr_dict_size = dictionary_sizes.sg
            D_init = model.sg.D{curr_dict_size};
            A = model.sg.A{curr_dict_size};
            B = model.sg.B{curr_dict_size};
            [D,A,B,~,~] = sg(train_data, D_init, params, A, B);
            model.sg.D{curr_dict_size} = D;
            model.sg.A{curr_dict_size} = A;
            model.sg.B{curr_dict_size} = B;
        end
    end
    %  
    if algorithms.neurogen_group_mairal
        for curr_dict_size = dictionary_sizes.neurogen_group_mairal
            D_init = model.neurogen_group_mairal.D{curr_dict_size};
            A = model.neurogen_group_mairal.A{curr_dict_size};
            B = model.neurogen_group_mairal.B{curr_dict_size};
            [D,A,B, ~, ~] = neurogen_group_mairal(train_data, D_init, params, A, B);
            model.neurogen_group_mairal.D{curr_dict_size} = D;
            model.neurogen_group_mairal.A{curr_dict_size} = A;
            model.neurogen_group_mairal.B{curr_dict_size} = B;
        end
    end
    %
    if algorithms.neurogen_mairal
        for curr_dict_size = dictionary_sizes.neurogen_mairal
            D_init = model.neurogen_mairal.D{curr_dict_size};
            A = model.neurogen_mairal.A{curr_dict_size};
            B = model.neurogen_mairal.B{curr_dict_size};
            [D,A,B, ~, ~] = neurogen_mairal(train_data, D_init, params, A, B);
            model.neurogen_mairal.D{curr_dict_size} = D;
            model.neurogen_mairal.A{curr_dict_size} = A;
            model.neurogen_mairal.B{curr_dict_size} = B;
        end
    end
    %
    if algorithms.neurogen_sg
        for curr_dict_size = dictionary_sizes.neurogen_sg
            D_init = model.neurogen_sg.D{curr_dict_size};
            A = model.neurogen_sg.A{curr_dict_size};
            B = model.neurogen_sg.B{curr_dict_size};
            [D,A,B, ~, ~] = neurogen_sg(train_data, D_init, params, A, B);
            model.neurogen_sg.D{curr_dict_size} = D;
            model.neurogen_sg.A{curr_dict_size} = A;
            model.neurogen_sg.B{curr_dict_size} = B;
        end
    end
end

function [D, A, B, error, correlation] = random_dummy(train_data, D_init, params, A, B)
    % random-D: just use the D_init
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for random case.\n');
    [D,A, B, error,correlation] = DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,-1,params.data_type,'Mairal', A, B);
end

function [D, A, B, error, correlation] =  neurogen_group_mairal(train_data, D_init, params, A, B)
    %neurogenesis - with GroupMairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Group Mairal.\n');
    [D, A, B,error,correlation] =  DL(train_data,D_init,params.nonzero_frac,params.lambda_D,params.mu,params.eta,params.epsilon,params.T,params.new_elements,params.data_type,'GroupMairal', A, B);
end

function [D, A, B, error, correlation] = neurogen_sg(train_data, D_init, params, A, B)
    % neurogenesis - with SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with SG.\n');
    [D, A, B, error,correlation] = DL(train_data,D_init,params.nonzero_frac,params.lambda_D,params.mu,params.eta,params.epsilon,params.T,params.new_elements,params.data_type,'SG', A, B);
end

function [D, A, B, error, correlation] = group_mairal(train_data, D_init, params, A, B)
    %%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly
    % group-sparse coding (Bengio et al 2009)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Group Mairal.\n');
    [D, A, B,error,correlation] = DL(train_data,D_init,params.nonzero_frac,params.lambda_D,params.mu,params.eta,params.epsilon,params.T,0,params.data_type,'GroupMairal', A, B);
end

function [D, A, B, error, correlation] = sg(train_data, D_init, params, A, B)
    % fixed-size-SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for SG (no sparsity though).\n');
    [D, A, B,error,correlation] = DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,0,params.data_type,'SG', A, B);
end

function [D, A, B, error, correlation] = mairal(train_data, D_init, params, A, B)
    % fixed-size-Mairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Mairal.\n');
    [D, A, B,error,correlation] = DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,0,params.data_type,'Mairal', A, B);
end

function [D, A, B, error, correlation] = neurogen_mairal(train_data, D_init, params, A, B)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Mairal.\n');
    [D, A, B, error,correlation] =  DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,params.new_elements,params.data_type,'Mairal', A, B);
end

function [train_data, test_data, data0, test_data0] = get_patches_data(T, dir_path)
    %real images (patches)
    [data0, test_data0] = boat_patches(T, dir_path);
    [train_data, test_data, ~] = lena_patches(T, dir_path);
end

function [train_data, test_data, data0, test_data0] = get_cifar_data(T, dir_path)
    [train_data_map, test_data_map, ~] = cifar_images_online(true, -1, dir_path);
    % sea images.
    train_data = train_data_map{72};
    test_data = test_data_map{72};
    assert (size(train_data, 2) == T);
    test_data0 = test_data_map{90};
    %     
    [train_data_map, ~, ~] = cifar_images_online(true, 100, dir_path);
    data0 = [];
    for curr_ns_label = 89:93
        data0 = [data0 train_data_map{curr_ns_label}];
    end
    assert (size(data0, 2) == T);
end

function [datasets_map] = get_datasets_map(is_patches, T, dir_path)
    if is_patches
        [train_data, test_data, data0, test_data0] = get_patches_data(T, dir_path);
    else
        [train_data, test_data, data0, test_data0] = get_cifar_data(T, dir_path);
    end
    %
    datasets_map = struct();
    datasets_map.data_st_train = train_data;
    datasets_map.data_st_test = test_data;
    datasets_map.data_nst_train = data0;
    datasets_map.data_nst_test = test_data0;
end

function [learned_dictionary_sizes, correlation, error] = post_process_results(evaluation, D, dictionary_sizes)
    learned_dictionary_sizes = [];
    correlation = [];
    error = [];
    % 
    curr_idx = 0;
    for curr_dict_size = dictionary_sizes
        curr_idx = curr_idx + 1;
        % 
        curr_learned_dict_size = size(D{curr_dict_size}, 2);
        learned_dictionary_sizes = [learned_dictionary_sizes; curr_learned_dict_size];
        %
        curr_evaluation = evaluation{curr_dict_size};
        % 
        curr_correlation = curr_evaluation.correlation;
        curr_correlation = curr_correlation(2, :);
        % 
        curr_error =  curr_evaluation.error;
        clear curr_evaluation;
        %
        error(curr_idx, :) = curr_error;
        correlation(curr_idx, :) = curr_correlation;
    end
end

function plot_correlation(model, dir_path)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    %     
    if model.algorithms.random
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'rs-');       
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'gv-');              
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'c+-');       
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);    
    curr_path = strcat(dir_path, sprintf('Figures/correlation_n%d_nz%d_T%d_new%d%s',params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

function plot_learned_dictionary_size(model, dir_path)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    % 
    if model.algorithms.random
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random);
        plot(model.dictionary_sizes.random, learned_dictionary_sizes,'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal);
        plot(model.dictionary_sizes.neurogen_group_mairal, learned_dictionary_sizes,'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg);
        plot(model.dictionary_sizes.neurogen_sg, learned_dictionary_sizes,'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal);
        plot(model.dictionary_sizes.group_mairal, learned_dictionary_sizes,'rs-');
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg);
        plot(model.dictionary_sizes.sg, learned_dictionary_sizes,'gv-');
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal);
        plot(model.dictionary_sizes.mairal, learned_dictionary_sizes,'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal);
        plot(model.dictionary_sizes.neurogen_mairal, learned_dictionary_sizes,'c+-');
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('initial dictionary size k');
    ylabel('learned dictionary size');
    curr_path = strcat(dir_path, sprintf('Figures/learnedk_n%d_nz%d_T%d_new%d%s',params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

