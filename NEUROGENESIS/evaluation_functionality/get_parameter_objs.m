function params_objs = get_parameter_objs()
    addpath('evaluation_functionality/');
    params = init_parameters();
    % 
    %params.n = 2500;
    count = 0;
    params_objs = {};
    % 
     for new_elements = [10 25]
        for lambda_D = [0.003 0.03]
%             for nz_in_dict = [0.0025 0.005 0.01 0.025 0.05]
                 for nonzero_frac = [0.0003 0.003]
                    count = count + 1;
                    %
                    curr_params_obj = params;
%                     curr_params_obj.new_elements = new_elements;
                    curr_params_obj.lambda_D = lambda_D;
%                     curr_params_obj.nz_in_dict = nz_in_dict;
%                     curr_params_obj.nonzero_frac = nonzero_frac;
                    %                 
                    params_objs{count} = curr_params_obj;
                 end
%             end
        end
     end
    %
    save('params_objs', 'params_objs');
end
