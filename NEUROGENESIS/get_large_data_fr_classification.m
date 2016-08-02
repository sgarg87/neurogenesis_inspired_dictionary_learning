function get_large_data_fr_classification()
    [flowers_data, oxford_data] = flower_building_images_online(1900, './', [100 100]);
    %     
    num_data_flowers = size(flowers_data, 2);
    num_data_oxford = size(oxford_data, 2);
    % 
    data = [flowers_data oxford_data];
    data = data';
    % 
    labels = [-1*ones(num_data_flowers, 1); ones(num_data_oxford, 1)];
    data = [data labels];
    %
    num_data = size(data, 1);
    rand_idx = randperm(num_data);
    data = data(rand_idx, :);
    %     
    save large_image_data data;
end
