function evaluation = evaluate_model(model, test_data)
    evaluation = struct();
    algorithms = model.algorithms;
    params = model.params;
    dictionary_sizes = model.dictionary_sizes;
    %     
    if algorithms.random
        evaluation.random = {};
        % 
        for curr_dict_size = dictionary_sizes.random
            D = model.random.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params);
            evaluation.random{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.mairal
        evaluation.mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.mairal
            D = model.mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params); 
            evaluation.mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.group_mairal
        evaluation.group_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.group_mairal
            D = model.group_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params);
            evaluation.group_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.sg
        evaluation.sg = {};
        % 
        for curr_dict_size = dictionary_sizes.sg
            D = model.sg.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params);
            evaluation.sg{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.neurogen_group_mairal
        evaluation.neurogen_group_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_group_mairal
            D = model.neurogen_group_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params);
            evaluation.neurogen_group_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.neurogen_sg
        evaluation.neurogen_sg = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_sg
            D = model.neurogen_sg.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params);
            evaluation.neurogen_sg{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
    %     
    if algorithms.neurogen_mairal
        evaluation.neurogen_mairal = {};
        % 
        for curr_dict_size = dictionary_sizes.neurogen_mairal
            D = model.neurogen_mairal.D{curr_dict_size};
            [~,error,correlation] = sparse_coding(test_data,D,params);
            evaluation.neurogen_mairal{curr_dict_size} = create_evaluation_object(error, correlation);
        end
    end
end

function curr_eval_obj = create_evaluation_object(error, correlation)
    curr_eval_obj = struct();
    curr_eval_obj.error = error;
    curr_eval_obj.correlation = correlation;
end
