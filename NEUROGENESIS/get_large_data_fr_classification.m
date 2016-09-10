function get_large_data_fr_classification(model)
    nst_test = model.datasets_map.data_nst_test;
    nst2_test = model.datasets_map.data_nst2_test;
    % 
    num_data_nst = size(nst_test, 2);
    num_data_nst2 = size(nst2_test, 2);
    % 
    data = [nst_test nst2_test];
    data = data';
    % 
    labels = [-1*ones(num_data_nst, 1); ones(num_data_nst2, 1)];
    data = [data labels];
    %
    num_data = size(data, 1);
    rand_idx = randperm(num_data);
    data = data(rand_idx, :);
    %     
    save large_image_data data;
end
