function [X] = voc_2012_images()   
    dir_path = './Data/VOCdevkit/VOC2012/JPEGImages/';
    all_jpeg_files = dir(strcat(dir_path, '*.jpg'));
%     
    all_jpeg_files_size = size(all_jpeg_files);
    num_data = all_jpeg_files_size(1);
    %   
    num_pixels_fr_selection = 100;
%     
    X = zeros(num_pixels_fr_selection, num_data);
%
    count = 0;
%     
    for curr_jpeg_file = all_jpeg_files'
        count = count + 1;
%         
        curr_file_path = strcat(dir_path, curr_jpeg_file.name);
        x = double(imread(curr_file_path))/255;
        % there seem to be three versions of an image. picking the first one. seems something specfic to jpg images.         
        x = x(:, :, 1);
        %
        num_pixels = prod(size(x));
        %
        rand_idx = randperm(num_pixels);
        rand_idx_sel = rand_idx(1:num_pixels_fr_selection);
        %         
        x = reshape(x, num_pixels, 1);
%         
%         size(x)
%         pcolor(x);
%         pause;
%         
        x_sel = x(rand_idx_sel);        
        X(:, count) = x_sel;
%         
%         if mod(count, 100) == 2
%             close;
%             pcolor(X);
%             pause;
%         end
%         
    end
    %
    clear x;
    %     
    % zero mean  
    X=X-repmat(mean(X),[size(X,1) 1]);
    % 1 std   
    X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
    