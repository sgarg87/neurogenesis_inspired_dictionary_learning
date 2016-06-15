function script_evaluate_online(is_hpcc)
    if ~is_hpcc
        addpath('evaluation_functionality/');
        addpath 'ElasticNet/';
    else
        addpath('evaluation_functionality/');
        addpath 'ElasticNet/';
    end
    % 
    evaluate_online(is_hpcc);
end
