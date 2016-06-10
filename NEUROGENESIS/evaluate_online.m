function evaluate_online()
    params = init_parameters();
    methods = get_list_of_methods_fr_experiments();
    datasets_map = get_datasets_map(params.is_patch);    
    %
    % todo: do it for multiple values of k appropriately.
    model = initialize_model(methods);
    model = adapt_model(model, datasets_map.data_nst_train, params);
    model.evaluation = evaluate_model(model, test_data, params);
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

function initialize_model(methods)    
    dictionary_sizes = get_dictionary_size_list_fr_methods(methods);
    assert(false);
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

function D_init = initialize_dictionary(n, k)
    rng(0);
    D_init = normalize(rand(n,k));
end

function curr_eval_obj = create_evaluation_object(error, correlation)
    curr_eval_obj = struct();
    curr_eval_obj.error = error;
    curr_eval_obj.correlation = correlation;
end

function evaluation = evaluate_model(model, test_data, params)
    evaluation = struct();
    methods = model.methods;
    %     
    if methods.random
        D = model.random.D;
        [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
        evaluation.random = create_evaluation_object(error, correlation);
    end
    %     
    if methods.mairal
        D = model.mairal.D;
        [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
        evaluation.mairal = create_evaluation_object(error, correlation);
    end
    %     
    if methods.group_mairal
        D = model.group_mairal.D;
        [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
        evaluation.group_mairal = create_evaluation_object(error, correlation);
    end
    %     
    if methods.sg
        D = model.sg.D;
        [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
        evaluation.sg = create_evaluation_object(error, correlation);
    end
    %     
    if methods.neurogen_group_mairal
        D = model.neurogen_group_mairal.D;
        [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
        evaluation.neurogen_group_mairal = create_evaluation_object(error, correlation);
    end
    %     
    if methods.neurogen_sg
        D = model.neurogen_sg.D;
        [~,error,correlation] = sparse_coding(test_data,D,params.nonzero_frac,params.data_type); % random-D
        evaluation.neurogen_sg = create_evaluation_object(error, correlation);
    end
end

function model = adapt_model(model, train_data, params)
    methods = model.methods;
    %
    if methods.random
        D_init = model.random.D;
        [D,~,~] = random(train_data, D_init, params);
        model.random.D = D;
    end
    %     
    if methods.mairal
        D_init = model.mairal.D;
        [D,~,~] = mairal(train_data, D_init, params);
        model.mairal.D = D;
    end
    %
    if methods.group_mairal
        D_init = model.group_mairal.D;
        [D,~,~] = group_mairal(train_data, D_init, params);
        model.group_mairal.D = D;
    end
    %     
    if methods.sg
        D_init = model.sg.D;
        [D,~,~] = sg(train_data, D_init, params);
        model.sg.D = D;
    end
    %  
    if methods.neurogen_group_mairal
        D_init = model.neurogen_group_mairal.D;
        [D, ~, ~] = neurogen_group_mairal(train_data, D_init, params);
        model.neurogen_group_mairal = D;
    end
    %
    if methods.neurogen_mairal
        D_init = model.neurogen_mairal.D;
        [D, ~, ~] = neurogen_mairal(train_data, D_init, params);
        model.neurogen_mairal = D;
    end
    %
    if methods.neurogen_sg
        D_init = model.neurogen_sg.D;
        [D, ~, ~] = neurogen_sg(train_data, D_init, params);
        model.neurogen_sg = D;
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

function plot_dictionary_learned_size()
    figure(1000+tt); hold on;
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

function plot_correlation()
    figure(tt+10000);
    errorbar(learned_k0,mean(correl0_P'),std(correl0_P'),'k--');
    hold on;        
    errorbar(learned_k1,mean(correl1_P'),std(correl1_P'),'bx-'); 
    errorbar(learned_k2,mean(correl2_P'),std(correl2_P'),'bo-');  
    errorbar(learned_k3,mean(correl3_P'),std(correl3_P'),'rs-'); 
    errorbar(learned_k4,mean(correl4_P'),std(correl4_P'),'gv-');
    errorbar(learned_k5,mean(correl5_P'),std(correl5_P'),'md-');
    errorbar(learned_k6,mean(correl6_P'),std(correl6_P'),'c+-');
    %
    legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal', 'neurogen-Mairal', 'location','SouthEast');
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    %
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);    
    saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
    saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    close(gcf);     
end

function plot_error()
    figure(tt+100);
    errorbar(learned_k0,mean(err0'),std(err0'),'k--'); 
    hold on;        
    errorbar(learned_k1,mean(err1'),std(err1'),'bx-');
    errorbar(learned_k2,mean(err2'),std(err2'),'bo-');  
    errorbar(learned_k3,mean(err3'),std(err3'),'rs-'); 
    errorbar(learned_k4,mean(err4'),std(err4'),'gv-');
    errorbar(learned_k5,mean(err5'),std(err5'),'md-');
    errorbar(learned_k6,mean(err6'),std(err6'),'c+-');
    %
    legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','neurogen-Mairal','location','SouthEast');
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    %
    xlabel('final dictionary size k');
    ylabel('MSE');
    ylim([0,1]);
    saveas(gcf,sprintf('Figures/%s_%s_err_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
    saveas(gcf,sprintf('Figures/%s_%s_err_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    close(gcf);
end

% function model = learn_model(train_data, D_init, methods, params)
%     model = struct();
%     model.methods = methods;
%     if methods.random
%         model.random = struct();        
%         [D,~,~] = random(train_data, D_init, params);
%         model.random.D = D;
%     end
%     %     
%     if methods.mairal
%         model.mairal = struct();        
%         [D,~,~] = mairal(train_data, D_init, params);
%         model.mairal.D = D;
%     end
%     %
%     if methods.group_mairal
%         model.group_mairal = struct();        
%         [D,~,~] = group_mairal(train_data, D_init, params);
%         model.group_mairal.D = D;
%     end
%     %     
%     if methods.sg
%         model.sg = struct();        
%         [D,~,~] = sg(train_data, D_init, params);
%         model.sg.D = D;
%     end
%     %  
%     if methods.neurogen_group_mairal
%         model.neurogen_group_mairal = struct();
%         [D, ~, ~] = neurogen_group_mairal(train_data, D_init, params);
%         model.neurogen_group_mairal = D;
%     end
%     %
%     if methods.neurogen_mairal
%         model.neurogen_mairal = struct();
%         [D, ~, ~] = neurogen_mairal(train_data, D_init, params);
%         model.neurogen_mairal = D;
%     end
%     %
%     if methods.neurogen_sg
%         model.neurogen_sg = struct();
%         [D, ~, ~] = neurogen_sg(train_data, D_init, params);
%         model.neurogen_sg = D;
%     end
% end
