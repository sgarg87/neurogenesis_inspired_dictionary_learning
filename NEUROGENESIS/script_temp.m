
[stationary_data_train, stationary_data_test, nonstationary_data_train, nonstationary_data_test, num_pixels] = cifar_images();
%
close;
% 
subplot(311);
pcolor(stationary_data_train(1:1024,1:2)');
subplot(312);
pcolor(stationary_data_train(1025:2048,1:2)');
subplot(313);
pcolor(stationary_data_train(2049:3072,1:2)');

