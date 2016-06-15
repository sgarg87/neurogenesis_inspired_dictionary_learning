function model = initialize_model(dir_path)
    model = struct();
    % 
    model.params = init_parameters();
    model.algorithms = get_list_of_algorithms_fr_experiments();
    model.datasets_map = get_datasets_map(model.params.is_patch, model.params.T, dir_path, model.params.n);
    %
    model.dictionary_sizes = get_dictionary_size_list_fr_algorithms(model.algorithms);
    rng(0);
    %     
    if model.algorithms.mairal
        curr_dictionary_sizes = model.dictionary_sizes.mairal;
        model.mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.random
        curr_dictionary_sizes = model.dictionary_sizes.random;
        model.random = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.group_mairal;
        model.group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.sg
        curr_dictionary_sizes = model.dictionary_sizes.sg;
        model.sg = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.neurogen_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_mairal;
        model.neurogen_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.neurogen_sg
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_sg;
        model.neurogen_sg = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
    %
    if model.algorithms.neurogen_group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_group_mairal;
        model.neurogen_group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params.n);
    end
end

function obj = initialize_D_A_B(curr_dictionary_sizes, n)
    D_init = {};
    A = {};
    B = {};
    for curr_k = curr_dictionary_sizes
        D_init{curr_k} = normalize(rand(n,curr_k));            
        A{curr_k} = [];
        B{curr_k} = [];
    end
    obj = struct();
    obj.D = D_init;
    obj.A = A;
    obj.B = B;
end
