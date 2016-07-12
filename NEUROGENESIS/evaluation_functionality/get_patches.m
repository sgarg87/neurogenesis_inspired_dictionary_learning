function patches_data = get_patches(datasets_map, patch_size)
    % putting all data together from different classes     
    all_data_train = [datasets_map.data_st_train datasets_map.data_nst_train];
    all_data_test = [datasets_map.data_st_test datasets_map.data_nst_test];
    clear dataset_map;
    %
    patches_data = struct();
    patches_data.train = compute_patches(all_data_train, patch_size);
    patches_data.test = compute_patches(all_data_test, patch_size);
end
