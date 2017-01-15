dir_path = './reconstructed images/';
% 
for curr_idx = 1:100
    model.params.is_sparse_data = 0;
    %     
    X = model.datasets_map.data_nst2_test(:, curr_idx); 
    imwrite(reshape(X, 100, 100), strcat(dir_path, strcat(num2str(curr_idx), 'org.png')));
%     imagesc(reshape(X, 100, 100)); saveas(gcf, strcat(dir_path, strcat(num2str(curr_idx), 'org_sc.png')));
    % 
    D = model.mairal.D{500};
    [C,~,~] = sparse_coding(X, D, model.params);
    X_ = D*C + mean(X);
    imwrite(reshape(X_, 100, 100), strcat(dir_path, strcat(num2str(curr_idx), 'mairal.png')));
%     imagesc(reshape(X_, 100, 100)); saveas(gcf, strcat(dir_path, strcat(num2str(curr_idx), 'mairal_sc.png')));
    % 
    D = model.neurogen_group_mairal.D{500};
    [C,~,~] = sparse_coding(X, D, model.params);
    X_ = D*C + mean(X);
    imwrite(reshape(X_, 100, 100), strcat(dir_path, strcat(num2str(curr_idx), 'ngmairal.png')));
%     imagesc(reshape(X_, 100, 100)); saveas(gcf, strcat(dir_path, strcat(num2str(curr_idx), 'ngmairal_sc.png')));
end

