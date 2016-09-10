function model = adapt_model(model, train_data)
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

function [D, A, B, error, correlation] = random_dummy(train_data, D_init, params, A, B)
    % random-D: just use the D_init
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for random case.\n');
    params.lambda_D = 0;
    params.new_elements = -1;
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'Mairal', A, B);
end

function [D, A, B, error, correlation] =  neurogen_group_mairal(train_data, D_init, params, A, B)
    %neurogenesis - with GroupMairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Group Mairal.\n');
    % 'GroupMairal'
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'GroupMairal', A, B);
end

function [D, A, B, error, correlation] = neurogen_sg(train_data, D_init, params, A, B)
    % neurogenesis - with SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with SG.\n');
    [D,A, B, error,correlation] = DL(train_data, D_init, params, 'SG', A, B);
end

function [D, A, B, error, correlation] = group_mairal(train_data, D_init, params, A, B)
    %%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly
    % group-sparse coding (Bengio et al 2009)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Group Mairal.\n');
    params.new_elements = 0;
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'GroupMairal', A, B);
end

function [D, A, B, error, correlation] = sg(train_data, D_init, params, A, B)
    % fixed-size-SG
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for SG (no sparsity though).\n');
    params.lambda_D = 0;
    params.new_elements = 0;
    [D, A, B, error,correlation] = DL(train_data,D_init, params, 'SG', A, B);
end

function [D, A, B, error, correlation] = mairal(train_data, D_init, params, A, B)
    % fixed-size-Mairal
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for Mairal.\n');
    params.lambda_D = 0;
    params.new_elements = 0;
    [D,A, B, error,correlation] = DL(train_data, D_init, params, 'Mairal', A, B);
end

function [D, A, B, error, correlation] = neurogen_mairal(train_data, D_init, params, A, B)
    fprintf('\n\n\n....................................')
    fprintf('Learning the dictionary model for neurogenesis with Mairal.\n');
    params.lambda_D = 0;
    [D,A, B, error,correlation] = DL(train_data,D_init, params, 'Mairal', A, B);
end
