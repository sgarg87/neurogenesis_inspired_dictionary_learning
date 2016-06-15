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
    plot_model_evaluation(model, dir_path);
    %
    save(strcat(dir_path, 'model'), 'model');
end
