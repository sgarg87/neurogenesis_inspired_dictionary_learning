function evaluate_online()
    params = init_parameters();
    methods = get_list_of_methods_fr_experiments();
    datasets_map = get_datasets_map(params.is_patch);
    model = initialize_model(methods);
    model.params = params;
    model.methods = methods;
    clear methods params;
    % 
    model = adapt_model(model, datasets_map.data_nst_train);
    model.evaluation = evaluate_model(model, test_data);
    % todo: generate plots.
    % todo: also save the model and the evaluation.
end

function methods = get_list_of_methods_fr_experiments()
    methods = struct();
    %     
    methods.mairal = false;
    methods.random = false;
    methods.group_mairal = false;
    methods.neurogen_group_mairal = false;
    methods.neurogen_mairal = false;
    methods.neurogen_sg = false;
    methods.sg = false;
    %     
    % setting it to false first and then true so that we can simply comment
    % the methods we don't want to use while running the experiments.     
    methods.mairal = true;
    methods.random = true;
    methods.group_mairal = true;
    methods.neurogen_group_mairal = true;
    methods.neurogen_mairal = true;
    methods.neurogen_sg = true;
    methods.sg = true;
end

function model = initialize_model(methods)
    dictionary_sizes = get_dictionary_size_list_fr_methods(methods);
    rng(0);
    %     
    if methods.mairal
        curr_dictionary_sizes = dictionary_sizes.mairal;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.mairal.D = D_init;
    end
    %
    if methods.random
        curr_dictionary_sizes = dictionary_sizes.random;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.random.D = D_init;
    end
    %
    if methods.group_mairal
        curr_dictionary_sizes = dictionary_sizes.group_mairal;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.group_mairal.D = D_init;
    end
    %
    if methods.sg
        curr_dictionary_sizes = dictionary_sizes.sg;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.sg.D = D_init;
    end
    %
    if methods.neurogen_mairal
        curr_dictionary_sizes = dictionary_sizes.neurogen_mairal;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.neurogen_mairal.D = D_init;
    end
    %
    if methods.neurogen_sg
        curr_dictionary_sizes = dictionary_sizes.neurogen_sg;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.neurogen_sg.D = D_init;
    end
    %
    if methods.neurogen_group_mairal
        curr_dictionary_sizes = dictionary_sizes.neurogen_group_mairal;
        D_init = {};
        for curr_k = curr_dictionary_sizes
            D_init{curr_k} = normalize(rand(n,curr_k));            
        end
        model.neurogen_group_mairal.D = D_init;
    end
    %
    model.dictionary_sizes = dictionary_sizes;
    model.methods = methods;
end

function dictionary_sizes = get_dictionary_size_list_fr_methods(methods)
    dictionary_sizes = struct();
    %
    if methods.mairal
        dictionary_sizes.mairal = [25 50 100 150];
    end
    %
    if methods.random
        dictionary_sizes.random = [25 50 100 150];
    end
    %
    if methods.group_mairal
        dictionary_sizes.group_mairal = [50 100 150 200 300];
    end
    %
    if methods.sg
        dictionary_sizes.sg = [25 50 100 150];
    end
    %
    if methods.neurogen_mairal
        dictionary_sizes.neurogen_mairal = [10 25 50 75];
    end
    %
    if methods.neurogen_sg
        dictionary_sizes.neurogen_sg = [10 25 50 75];
    end
    %
    if methods.neurogen_group_mairal
        dictionary_sizes.neurogen_group_mairal = [25 50 100 150];
    end 
end

function curr_eval_obj = create_evaluation_object(error, correlation)
    curr_eval_obj = struct();
    curr_eval_obj.error = error;
    curr_eval_obj.correlation = correlation;
end

function evaluation = evaluate_model(model, test_data, params)
    evaluation = struct();
    methods = model.methods;
    dictionary_sizes = model.dictionary_sizes;
    %     
    if methods.random
        evaluation.random = {};
        % 
        for curr_dict_size = dictionary_sizes.random
            D = model.random.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.random{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if methods.mairal
        evaluation.mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.mairal
            D = model.mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if methods.group_mairal
        evaluation.group_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.group_mairal
            D = model.group_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.group_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if methods.sg
        evaluation.sg = {};
        % 
        for curr_dict_size = dictionary_sizes.sg
            D = model.sg.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.sg{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if methods.neurogen_group_mairal
        evaluation.neurogen_group_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_group_mairal
            D = model.neurogen_group_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.neurogen_group_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if methods.neurogen_sg
        evaluation.neurogen_sg = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_sg
            D = model.neurogen_sg.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
            evaluation.neurogen_sg{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
end

function model = adapt_model(model, train_data)
    methods = model.methods;
    dictionary_sizes = model.dictionary_sizes;
    params = model.params;
    %
    if methods.random
        for curr_dict_size = dictionary_sizes.random
            D_init = model.random.D{curr_dict_size};
            [D,~,~] = random(train_data, D_init, params);
            model.random.D{curr_dict_size} = D;            
        end
    end
    %     
    if methods.mairal
        for curr_dict_size = dictionary_sizes.mairal
            D_init = model.mairal.D{curr_dict_size};
            [D,~,~] = mairal(train_data, D_init, params);
            model.mairal.D{curr_dict_size} = D;
        end
    end
    %
    if methods.group_mairal
        for curr_dict_size = dictionary_sizes.group_mairal
            D_init = model.group_mairal.D{curr_dict_size};
            [D,~,~] = group_mairal(train_data, D_init, params);
            model.group_mairal.D{curr_dict_size} = D;
        end
    end
    %     
    if methods.sg
        for curr_dict_size = dictionary_sizes.sg
            D_init = model.sg.D{curr_dict_size};
            [D,~,~] = sg(train_data, D_init, params);
            model.sg.D{curr_dict_size} = D;
        end
    end
    %  
    if methods.neurogen_group_mairal
        for curr_dict_size = dictionary_sizes.neurogen_group_mairal
            D_init = model.neurogen_group_mairal.D{curr_dict_size};
            [D, ~, ~] = neurogen_group_mairal(train_data, D_init, params);
            model.neurogen_group_mairal.D{curr_dict_size} = D;
        end
    end
    %
    if methods.neurogen_mairal
        for curr_dict_size = dictionary_sizes.neurogen_mairal
            D_init = model.neurogen_mairal.D{curr_dict_size};
            [D, ~, ~] = neurogen_mairal(train_data, D_init, params);
            model.neurogen_mairal.D{curr_dict_size} = D;
        end
    end
    %
    if methods.neurogen_sg
        for curr_dict_size = dictionary_sizes.neurogen_sg
            D_init = model.neurogen_sg.D{curr_dict_size};
            [D, ~, ~] = neurogen_sg(train_data, D_init, params);
            model.neurogen_sg.D{curr_dict_size} = D;
        end
    end
end

function [D,error,correlation] = random(train_data, D_init, params)
    % random-D: just use the D_init
    % sahil: D_update_method value doesn't seem to be random in the scripts
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for random case.\n');
    [D,error,correlation] = DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,-1,params.data_type,'Mairal');
end

function [D,error,correlation] =  neurogen_group_mairal(train_data, D_init, params)
    %neurogenesis - with GroupMairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Group Mairal.\n');
    [D,error,correlation] =  DL(train_data,D_init,params.nonzero_frac,params.lambda_D,params.mu,params.eta,params.epsilon,params.T,params.new_elements,params.data_type,'GroupMairal');
end

function [D, error,correlation] = neurogen_sg(train_data, D_init, params)
    % neurogenesis - with SG
    %[D2,err22,correl22] = DL_neurogen(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with SG.\n');
    [D, error,correlation] = DL(train_data,D_init,params.nonzero_frac,params.lambda_D,params.mu,params.eta,params.epsilon,params.T,params.new_elements,params.data_type,'SG');
end

function [D,error,correlation] = group_mairal(train_data, D_init, params)
    %%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly
    % group-sparse coding (Bengio et al 2009)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Group Mairal.\n');
    [D,error,correlation] = DL(train_data,D_init,params.nonzero_frac,params.lambda_D,params.mu,params.eta,params.epsilon,params.T,0,params.data_type,'GroupMairal');
end

function [D,error,correlation] = sg(train_data, D_init, params)
    % fixed-size-SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for SG (no sparsity though).\n');
    [D,error,correlation] = DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,0,params.data_type,'SG');
end

function [D,error,correlation] = mairal(train_data, D_init, params)
    % fixed-size-Mairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Mairal.\n');
    [D,error,correlation] = DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,0,params.data_type,'Mairal');
end

function [D,error,correlation] = neurogen_mairal(train_data, D_init, params)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Mairal.\n');
    [D,error,correlation] =  DL(train_data,D_init,params.nonzero_frac,0,params.mu,params.eta,params.epsilon,params.T,params.new_elements,params.data_type,'Mairal');
end

function [train_data, test_data, data0, test_data0] = get_patches_data()
    %real images (patches)
    [data0, test_data0] = boat_patches(T);
    [train_data, test_data, ~] = lena_patches(T);
end

function [train_data, test_data, data0, test_data0] = get_cifar_data()
    [train_data_map, test_data_map, ~] = cifar_images_online(true, -1);
    % sea images.
    train_data = train_data_map{72};
    test_data = test_data_map{72};
    assert (size(train_data, 2) == T);
    test_data0 = test_data_map{90};
    %     
    [train_data_map, ~, ~] = cifar_images_online(true, 100);
    data0 = [];
    for curr_ns_label = 89:93
        data0 = [data0 train_data_map{curr_ns_label}];
    end
    assert (size(data0, 2) == T);
end

function [datasets_map] = get_datasets_map(is_patches)
    if is_patches
        [train_data, test_data, data0, test_data0] = get_patches_data();
    else
        [train_data, test_data, data0, test_data0] = get_cifar_data();
    end
    %
    datasets_map = struct();
    datasets_map.data_st_train = train_data;
    datasets_map.data_st_test = test_data;
    datasets_map.data_nst_train = data0;
    datasets_map.data_nst_test = test_data0;
end

function params = init_parameters()
    params = struct();
    params.eta = 0.1;
    params.adapt='basic'; %'adapt';
    params.test_or_train = 'nonstat';
    params.T = 100;
    params.k_array = [5 10 15 20 25 75 100];
    params.new_elements = 1;
    params.epsilon = 1e-2;
    params.lambda_D = 0.03;
    params.mu = 0;
    params.data_type = 'Gaussian';
    params.noise = 5;
    params.True_nonzero_frac = 0.2;
    params.nonzero_frac = 0.05;
    params.test_or_train = 'train';
    params.is_patch = false;
end

function plot_dictionary_learned_size(model)    
    close;
    plot(k_array,learned_k0,'k--',k_array,learned_k1,'bx-',k_array,learned_k2,'bo-',k_array,learned_k3,'rs-',...
        k_array,learned_k4,'gv-',k_array,learned_k5,'md-', k_array,learned_k6,'c+-');
    %     
    legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal', 'neurogen-Mairal', 'location','SouthEast'); 
    %     
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    xlabel('initial dictionary size k');
    ylabel('learned dictionary size');
    % 
    saveas(gcf,sprintf('Figures/%s_%s_learn_k_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
    saveas(gcf,sprintf('Figures/%s_%s_learn_k_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    close(gcf);
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
        learned_dictionary_sizes = [learned_dictionary_sizes; curr_learned_dict_size]
        %
        curr_evaluation = evaluation{curr_dict_size}
        curr_correlation = curr_evaluation.correlation;
        curr_error =  curr_evaluation.error;
        clear curr_evaluation;
        %
        error(curr_idx, :) = curr_error;
        correlation(curr_idx, :) = curr_correlation;
    end
end

function plot_correlation(model)
    params = model.params;
    % 
    figure(tt+10000);
    hold on;        
    % 
    if model.methods.random
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'k--', label='random-D');
    end
    % 
    if model.methods.neurogen_group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'bx-', label='neurogen-groupMairal');
    end
    % 
    if model.methods.neurogen_sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'bo-', label='neurogen-SG');
    end
    % 
    if model.methods.group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'rs-', label='groupMairal');       
    end
    % 
    if model.methods.sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'gv-', label='SG');              
    end
    % 
    if model.methods.mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'md-', label='Mairal');
    end
    % 
    if model.methods.neurogen_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'c+-', label='neurogen-Mairal');        
    end
    % 
    legend('location','SouthEast');
    %
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);    
    saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
    saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    close(gcf);     
end

