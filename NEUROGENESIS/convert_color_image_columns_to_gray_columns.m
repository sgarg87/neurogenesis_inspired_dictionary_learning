function [data_gray] = convert_color_image_columns_to_gray_columns(data, dim1, dim2)
    % assuming that each column of X is a single data point.
    num_data = size(data, 2);
    num_pixels = dim1*dim2;
    data_gray = -1*ones(num_pixels, num_data);
%     
    for curr_data_idx = 1:num_data
        curr_data_vector = data(:, curr_data_idx);
        curr_data = reshape(curr_data_vector, dim1, dim2, 3);
        clear curr_data_vector;
%         
        curr_data_gray = rgb2gray(curr_data);
        clear curr_data;
%         
        curr_data_gray_vector = reshape(curr_data_gray, num_pixels, 1);
        clear curr_data_gray;
%         
        data_gray(:, curr_data_idx) = curr_data_gray_vector;
        clear curr_data_gray_vector;
    end    
end
