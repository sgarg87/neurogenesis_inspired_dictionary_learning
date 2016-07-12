function [ correlation ] = evaluate_patches_dictionary(patches_D, patches_data_test, patches_params)
    [~,~,correlation] = sparse_coding(patches_data_test, patches_D, patches_params); 
end
