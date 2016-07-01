function model = initialize_model(dir_path)
    model = struct();
    %
    % setting seed zero for all random samplings.     
%     rng(0);
    %     
    model.params = init_parameters();
    model.algorithms = get_list_of_algorithms_fr_experiments();
    model.datasets_map = get_datasets_map(model.params.data_set_name, model.params.T, dir_path, model.params.n);
    %
    model.dictionary_sizes = get_dictionary_size_list_fr_algorithms(model.algorithms);
    %     
    if model.algorithms.mairal
        curr_dictionary_sizes = model.dictionary_sizes.mairal;
        model.mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.random
        curr_dictionary_sizes = model.dictionary_sizes.random;
        model.random = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.group_mairal;
        model.group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.sg
        curr_dictionary_sizes = model.dictionary_sizes.sg;
        model.sg = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.neurogen_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_mairal;
        model.neurogen_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.neurogen_sg
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_sg;
        model.neurogen_sg = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
    %
    if model.algorithms.neurogen_group_mairal
        curr_dictionary_sizes = model.dictionary_sizes.neurogen_group_mairal;
        model.neurogen_group_mairal = initialize_D_A_B(curr_dictionary_sizes, model.params);
    end
end

function obj = initialize_D_A_B(curr_dictionary_sizes, params)
    D_init = {};
    A = {};
    B = {};
    for curr_k = curr_dictionary_sizes
        if params.is_sparse_dict_init
            warning('sparse initializations of dictionary elements are not normalized.');
            D_init{curr_k} = sprand(params.n,curr_k, params.nz_in_dict);
        else
            D_init{curr_k} = normalize(rand(params.n,curr_k));
        end
        assert(~nnz(isnan(D_init{curr_k})));
        %
        if params.is_init_A
            if params.is_A_sparse
                curr_random_init = sprand(curr_k, curr_k, params.A_sparse_nnz);
                curr_random_init = curr_random_init + diag(rand(curr_k, 1));
            else
                curr_random_init = rand(curr_k, curr_k);
            end
            %             
            A{curr_k} = params.A*(curr_random_init'*curr_random_init);
        else
            A{curr_k} = [];
        end
        %
        if params.is_init_B
            error('Not implemented in conformance with A. Also, sparse case not implemented yet');
            B{curr_k} = params.B*rand(params.n, curr_k);
        else
            B{curr_k} = [];
        end
    end
    obj = struct();
    obj.D = D_init;
    obj.A = A;
    obj.B = B;
end
