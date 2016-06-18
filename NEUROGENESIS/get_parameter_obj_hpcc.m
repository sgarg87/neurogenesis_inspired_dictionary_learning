function params = get_parameter_obj_hpcc()
    addpath('evaluation_functionality/');
    params = init_parameters();
    params.n = 625;
    params.new_elements = 250;
    params.lambda_D = 0.003;
    %
    save('hpcc_param_obj', 'params');
end
