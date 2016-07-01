function params = get_parameter_obj_hpcc()
    addpath('evaluation_functionality/');
    params = init_parameters();
%     params.n = 256;
%     params.lambda_D = 0.3;
    %
    save('hpcc_param_obj', 'params');
end
