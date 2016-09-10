function [flowers_data, animals_data, oxford_data] = flower_building_images_online(num_data_per_label, dir_path, image_size)
%     flower_dir_path = strcat(dir_path, '../../data/image data in use/data set 1/all_data_sets/');
%     flower_dir_path = strcat(dir_path, '../../data/images_flowers_dogs_cats/');
    flower_dir_path = strcat(dir_path, '../../data/102flowers/');
    flower_image_files = dir(strcat(flower_dir_path, '*.jpg'));
    flower_image_files = flower_image_files(randperm(length(flower_image_files)));
    flower_image_files = flower_image_files(1:num_data_per_label);
    flowers_data = process_image_files(flower_dir_path, flower_image_files, image_size, dir_path);
    clear flower_dir_path flower_image_files;
%     flowers_data = add_noise(flowers_data);
    %
    animals_dir_path = strcat(dir_path, '../../data/images_dogs_cats/');
    animal_image_files = dir(strcat(animals_dir_path, '*.jpg'));
    animal_image_files = animal_image_files(randperm(length(animal_image_files)));
    animal_image_files = animal_image_files(1:num_data_per_label);
    animals_data = process_image_files(animals_dir_path, animal_image_files, image_size, dir_path);
    clear animals_dir_path animal_image_files;
%     animals_data = add_noise(animals_data);
    %
%     oxford_dir_path = strcat(dir_path, '../../data/image data in use/data set 2/all_data_sets/');
    oxford_dir_path = strcat(dir_path, '../../data/oxbuild_images/');
    oxford_image_files = dir(strcat(oxford_dir_path, '*.jpg'));
    oxford_image_files = oxford_image_files(randperm(length(oxford_image_files)));
    oxford_image_files = oxford_image_files(1:num_data_per_label);
    oxford_data = process_image_files(oxford_dir_path, oxford_image_files, image_size, dir_path);
    oxford_data = oxford_data(:, 1:num_data_per_label);
    clear oxford_dir_path oxford_image_files;
%     oxford_data = add_noise(oxford_data);
end

function y_new = add_noise(y)
    y_new = y + (min(min(abs(y)))*1e-1)*rand(size(y));
end

function data = process_image_files(image_dir, image_files_list, image_size, dir_path)
    num_images = length(image_files_list);
    data_dim = prod(image_size);
    %
    data = zeros(data_dim, num_images);
    %
    curr_idx = 0;
    for curr_image_file = image_files_list'
        curr_idx = curr_idx + 1;
        curr_data = process_image(strcat(image_dir, curr_image_file.name), image_size, dir_path);
        data(:, curr_idx) = curr_data; clear curr_data;
    end
end

function I = process_image(file_path, image_size, dir_path)
    image_dim = prod(image_size);
    %
    I = imread(file_path);
    I = imresize(I, image_size);    
%     
    if size(I, 3) == 3
        I = rgb2gray(I);
    end
%     
    I = double(I)/255; 
    imwrite(I, strcat(dir_path, './temp.png'));
    %     
%     H = fspecial('unsharp');
%     I = imfilter(I, H);
    %
    I = reshape(I, image_dim, 1);
    %     
%     I = preprocess_data(I);
%     imwrite(reshape(I, image_size)*255, strcat(dir_path, './temp_preprocessed.png'));
    %
    assert(~nnz(isnan(I)));
    assert(nnz(I) ~= 0);
end
    