function patches_data = compute_patches(all_data, patch_size)
    % generating patches from the images
    num_data = size(all_data, 2);
    dimension = size(all_data, 1);
    %
    image_dim = sqrt(dimension);
    assert(mod(image_dim, 1) == 0);
    %
    patches_data = [];
    %     
    for curr_idx = 1:num_data
        curr_image_data = all_data(:, curr_idx);
        curr_image_data = reshape(curr_image_data, image_dim, image_dim);
        %         
        curr_patches_data = im2col(curr_image_data, patch_size, 'distinct');
        %
        if curr_idx == 1
            dim_patch = size(curr_patches_data, 1);
            num_patches_per_data = size(curr_patches_data, 2);
            patches_data = zeros(dim_patch, num_data*num_patches_per_data);
        end
        %         
        curr_patches_range = ((curr_idx-1)*num_patches_per_data+1):(curr_idx*num_patches_per_data);
        patches_data(:, curr_patches_range) = curr_patches_data;
    end
end
