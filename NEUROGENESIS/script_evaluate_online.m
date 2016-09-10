function script_evaluate_online(is_hpcc, curr_core)
    addpath('evaluation_functionality/');
    addpath 'ElasticNet/';
    %
    evaluate_online(is_hpcc, curr_core);
end
