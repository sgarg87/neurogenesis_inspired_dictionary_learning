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
            D_init{curr_k}
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