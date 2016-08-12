% This script is written by Sahil Garg, consolidating the functionality of
% the original scripts main_online and run_experiments.
% This is more flexible in terms of choosing which algorithms to evaluate
% in the simulations (experiments).
% 
function evaluate_online(is_hpcc, curr_core)
    dir_path = get_dir_path(is_hpcc);
    % 
    model = initialize_model(dir_path);
    %
    if is_hpcc && (curr_core > 0)
        params_objs = get_parameter_objs();
        model.params = params_objs{curr_core};
    elseif is_hpcc
        model.params = get_parameter_obj_hpcc();
    end
    %
    %
    model = adapt_model(model, model.datasets_map.data_st_train);
    save(strcat(dir_path, 'old_model'), 'model');
    %     
    if isfield(model.datasets_map, 'data_nst2_train')
        all_train_data = [model.datasets_map.data_nst_train model.datasets_map.data_nst2_train];
        num_all_train_data = size(all_train_data, 2);
        all_train_rnd_idx = randperm(num_all_train_data);
        all_train_data = all_train_data(:, all_train_rnd_idx);
        clear all_train_rnd_idx;
        model = adapt_model(model, all_train_data);
        clear all_train_data;
    else
        model = adapt_model(model, model.datasets_map.data_nst_train);
    end
    %     
    % replay
    if model.params.is_replay
        model = adapt_model(model, model.datasets_map.data_st_train);
        model = adapt_model(model, model.datasets_map.data_nst_train);
    end
    %    
    model.evaluation_st = evaluate_model(model, model.datasets_map.data_st_test);
    model.evaluation = model.evaluation_st;
    plot_model_evaluation(model, dir_path, 'first_domain');
    %     
    model.evaluation_nst = evaluate_model(model, model.datasets_map.data_nst_test);
    model.evaluation = model.evaluation_nst;
    plot_model_evaluation(model, dir_path, 'second_domain');
    %
    if isfield(model.datasets_map, 'data_nst2_test')
        model.evaluation_nst2 = evaluate_model(model, model.datasets_map.data_nst2_test);
        model.evaluation = model.evaluation_nst2;
        plot_model_evaluation(model, dir_path, 'third_domain');
    end
    %     
    model_path = 'model';
    if is_hpcc && (curr_core > 0)
        model_path = strcat(model_path, num2str(curr_core));        
    end
    save(strcat(dir_path, model_path), 'model');
end
