function [ train_data_1, test_data_1, train_data_2, test_data_2 ] = synthetic_sparse_data(input_dim, T1, T2)
    ratio_num_nonzeros = 0.05;
    ratio_num_nonzeros_off_diagonal = 0.001;
    %     
    num_nonzeros = floor(ratio_num_nonzeros*input_dim); clear ratio_num_nonzeros;
    num_nonzeros_off_diagonal = floor(ratio_num_nonzeros_off_diagonal*input_dim); clear ratio_num_nonzeros_off_diagonal;
    %
    % train_data_1
    train_data_1 = zeros(input_dim, T1);
    train_data_1(1:input_dim/2, :) = sample_sparse_data(T1, input_dim/2, num_nonzeros);
    if num_nonzeros_off_diagonal > 0
        train_data_1(input_dim/2+1:end, :) = sample_sparse_data(T1, input_dim/2, num_nonzeros_off_diagonal);
    end
    %
    test_data_1 = zeros(input_dim, T1);
    test_data_1(1:input_dim/2, :) = sample_sparse_data(T1, input_dim/2, num_nonzeros);
    if num_nonzeros_off_diagonal > 0
        test_data_1(input_dim/2+1:end, :) = sample_sparse_data(T1, input_dim/2, num_nonzeros_off_diagonal);
    end
    %
    train_data_2 = zeros(input_dim, T2); 
    train_data_2(input_dim/2+1:end, :) = sample_sparse_data(T2, input_dim/2, num_nonzeros);
    if num_nonzeros_off_diagonal > 0
        train_data_2(1:input_dim/2, :) = sample_sparse_data(T2, input_dim/2, num_nonzeros_off_diagonal);
    end
    %
    test_data_2 = zeros(input_dim, T2);
    test_data_2(input_dim/2+1:end, :) = sample_sparse_data(T2, input_dim/2, num_nonzeros);
    if num_nonzeros_off_diagonal > 0
        test_data_2(1:input_dim/2, :) = sample_sparse_data(T2, input_dim/2, num_nonzeros_off_diagonal);
    end
end

function data = sample_sparse_data(T, input_dim, num_nonzeros)
    data = zeros(input_dim, T);
    for curr_idx = 1:T
        curr_nonzero_idx = randperm(input_dim, num_nonzeros);
        data(curr_nonzero_idx, curr_idx) = 1;
    end
end
