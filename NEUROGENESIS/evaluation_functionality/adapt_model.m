function model = adapt_model(model, train_data)
    % 
    algorithms = model.algorithms;
    dictionary_sizes = model.dictionary_sizes;
    params = model.params;
    %
    if algorithms.neurogen_group_mairal
        for curr_dict_size = dictionary_sizes.neurogen_group_mairal
            D_init = model.neurogen_group_mairal.D{curr_dict_size};
            A = model.neurogen_group_mairal.A{curr_dict_size};
            B = model.neurogen_group_mairal.B{curr_dict_size};
            [D,A,B, ~, ~] = neurogen_group_mairal(train_data, D_init, params, A, B);
            model.neurogen_group_mairal.D{curr_dict_size} = D;
            model.neurogen_group_mairal.A{curr_dict_size} = A;
            model.neurogen_group_mairal.B{curr_dict_size} = B;
        end
    end
    %
    if algorithms.mairal
        for curr_dict_size = dictionary_sizes.mairal
            D_init = model.mairal.D{curr_dict_size};
            A = model.mairal.A{curr_dict_size};
            B = model.mairal.B{curr_dict_size};
            [D,A,B,~,~] = mairal(train_data, D_init, params, A, B);
            model.mairal.D{curr_dict_size} = D;
            model.mairal.A{curr_dict_size} = A;
            model.mairal.B{curr_dict_size} = B;
        end
    end
    %
    if algorithms.random
        for curr_dict_size = dictionary_sizes.random
            D_init = model.random.D{curr_dict_size};
            A = model.random.A{curr_dict_size};
            B = model.random.B{curr_dict_size};
            [D,A,B, ~,~] = random_dummy(train_data, D_init, params, A, B);
            model.random.D{curr_dict_size} = D;
            model.random.A{curr_dict_size} = A;
            model.random.B{curr_dict_size} = B;
        end
    end
    %     
    if algorithms.group_mairal
        for curr_dict_size = dictionary_sizes.group_mairal
            D_init = model.group_mairal.D{curr_dict_size};
            A = model.group_mairal.A{curr_dict_size};
            B = model.group_mairal.B{curr_dict_size};
            [D,A,B,~,~] = group_mairal(train_data, D_init, params, A, B);
            model.group_mairal.D{curr_dict_size} = D;
            model.group_mairal.A{curr_dict_size} = A;
            model.group_mairal.B{curr_dict_size} = B;
        end
    end
    %     
    if algorithms.sg
        for curr_dict_size = dictionary_sizes.sg
            D_init = model.sg.D{curr_dict_size};
            A = model.sg.A{curr_dict_size};
            B = model.sg.B{curr_dict_size};
            [D,A,B,~,~] = sg(train_data, D_init, params, A, B);
            model.sg.D{curr_dict_size} = D;
            model.sg.A{curr_dict_size} = A;
            model.sg.B{curr_dict_size} = B;
        end
    end
    %  
    if algorithms.neurogen_mairal
        for curr_dict_size = dictionary_sizes.neurogen_mairal
            D_init = model.neurogen_mairal.D{curr_dict_size};
            A = model.neurogen_mairal.A{curr_dict_size};
            B = model.neurogen_mairal.B{curr_dict_size};
            [D,A,B, ~, ~] = neurogen_mairal(train_data, D_init, params, A, B);
            model.neurogen_mairal.D{curr_dict_size} = D;
            model.neurogen_mairal.A{curr_dict_size} = A;
            model.neurogen_mairal.B{curr_dict_size} = B;
        end
    end
    %
    if algorithms.neurogen_sg
        for curr_dict_size = dictionary_sizes.neurogen_sg
            D_init = model.neurogen_sg.D{curr_dict_size};
            A = model.neurogen_sg.A{curr_dict_size};
            B = model.neurogen_sg.B{curr_dict_size};
            [D,A,B, ~, ~] = neurogen_sg(train_data, D_init, params, A, B);
            model.neurogen_sg.D{curr_dict_size} = D;
            model.neurogen_sg.A{curr_dict_size} = A;
            model.neurogen_sg.B{curr_dict_size} = B;
        end
    end
end

