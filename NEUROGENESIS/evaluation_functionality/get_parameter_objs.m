function params_objs = get_parameter_objs()
    addpath('evaluation_functionality/');
    params = init_parameters();
    % 
    count = 0;
    params_objs = {};
    % 
            for nz_in_dict = [0.005 0.020, 0.050]
                    count = count + 1;
                    %
                    curr_params_obj = params;
                    curr_params_obj.nz_in_dict = nz_in_dict;
                    %                 
                    params_objs{count} = curr_params_obj;
            end
    %
    save('params_objs', 'params_objs');
end
