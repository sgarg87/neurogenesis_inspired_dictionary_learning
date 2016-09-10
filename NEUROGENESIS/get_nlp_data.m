function [ data1, data2 ] = get_nlp_data(num_data_per_class, dir_path, input_dim)
    file_path = strcat(dir_path, 'data_adjacency_matrix.mat');
    load(file_path);
    %
    data_adjacency_matrix = data_adjacency_matrix(1:input_dim, :);
    %     
    data1 = data_adjacency_matrix(:, 1:num_data_per_class);
    data2 = data_adjacency_matrix(:, num_data_per_class+1:num_data_per_class*2);
end
