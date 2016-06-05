clear;
close all;
addpath './ElasticNet';
% 
% 
k = 300;
nonzero_frac = 0.05;
mu = 0;
eta = 0.1;
epsilon = 1e-2;
T = 300;
data_type = 'Gaussian';
% 
% each column is a data point. 
[train_data_stationary, ~, ~, test_data_nonstationary, n] = cifar_images(true);
train_data_stationary = train_data_stationary(:, 1:T); test_data_nonstationary = test_data_nonstationary(:, 1:T);
% 
D_init = normalize(rand(n,k));
% 
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for Mairal.\n');
[D, error_train, correlation_train] = DL(train_data_stationary, D_init,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'Mairal');
%
[C, error_test, correlation_test] = sparse_coding(test_data_nonstationary, D,nonzero_frac,data_type);    % Mairal  
% 
% display(C);
% 
spy(C);
xlabel('Data Points');
ylabel('Dictionary elements');
saveas(gcf, './temp_fig/coefficients.png','png');
close(gcf);
