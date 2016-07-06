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
    model = adapt_model(model, model.datasets_map.data_st_train);
    save(strcat(dir_path, 'old_model'), 'model');
    %     
    model = adapt_model(model, model.datasets_map.data_nst_train);
    %     
    model.evaluation = evaluate_model(model);
    %
    plot_model_evaluation(model, dir_path);
    %
    model_path = 'model';
    if is_hpcc && (curr_core > 0)
        model_path = strcat(model_path, num2str(curr_core));        
    end
    save(strcat(dir_path, model_path), 'model');
end

