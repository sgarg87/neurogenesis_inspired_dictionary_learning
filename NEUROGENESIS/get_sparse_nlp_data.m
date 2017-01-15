function [ train_data_1, test_data_1, train_data_2, test_data_2 ] = get_sparse_nlp_data(is_random_shuffle)
    if is_random_shuffle
        display('warning: randomly shuffling data');
    end
% 
% 
% 
    load('data_adjacency_matrix_selected_ml_it_nu.mat');
    data_adjacency_matrix = data_adjacency_matrix_selected;
    clear data_adjacency_matrix_selected;
%     
% 
    if is_random_shuffle
        data_adjacency_matrix = data_adjacency_matrix(:,randperm(size(data_adjacency_matrix, 2)));
    end
% 
% 
    data2 = data_adjacency_matrix(:, 1:5500);
    data1 = data_adjacency_matrix(:, 5501:end);
    %     
    train_data_1 = data1(:, 1:2:5500);
    test_data_1 = data1(:, 2:2:5500);
    clear data1;
    %     
    train_data_2 = data2(:, 1:2:5500);
    test_data_2 = data2(:, 2:2:5500);
    clear data2;
% 
% 
% 
    %     
%     data_adjacency_matrix = data_adjacency_matrix(1:3:3*5000, :);
%     data_adjacency_matrix = data_adjacency_matrix(1:16:16*1024, :);
    %
%     data_adjacency_matrix = data_adjacency_matrix(:, sum(data_adjacency_matrix) ~= 0);
%     %     
%     data2 = data_adjacency_matrix(:, 1:6000);
%     data1 = data_adjacency_matrix(:, 6001:12000);
%     %
%     train_data_1 = data1(:, 1:20:end);
%     test_data_1 = data1(:, 2:20:end);
%     clear data1;
%     %     
%     train_data_2 = data2(:, 1:20:end);
%     test_data_2 = data2(:, 2:20:end);
%     clear data2;
end
