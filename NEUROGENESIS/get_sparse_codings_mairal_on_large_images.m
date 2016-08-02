function data = get_sparse_codings_mairal_on_large_images(model, dict_size_org)
    % 
    [C_st_test,~,~] = sparse_coding(model.datasets_map.data_st_test, model.mairal.D{dict_size_org}, model.params);
    [C_nst_test,~,~] = sparse_coding(model.datasets_map.data_nst_test, model.mairal.D{dict_size_org}, model.params);
    % 
    num_data_st = size(C_st_test, 2);
    num_data_nst = size(C_nst_test, 2);
    % 
    data = [C_st_test C_nst_test];
    data = data';
    % 
    labels = [-1*ones(num_data_st, 1); ones(num_data_nst, 1)];
    data = [data labels];
    %
    num_data = size(data, 1);
    rand_idx = randperm(num_data);
    data = data(rand_idx, :);
    %     
    save mairal_sparse_codings_test data;
end
