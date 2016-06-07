function [new_elements,err0, correl0, learned_k0, err1, correl1, learned_k1, ...
    err2, correl2,learned_k2,err3, correl3, learned_k3, ...
    err4, correl4, learned_k4, err5, correl5, learned_k5] = run_experiments(params)
% 
% what is n, k, t
n = params.n;
k = params.k;
T = params.T;
% 
fprintf('\n\n\n\n\n')
fprintf('************************No. of dictionary elements to start is %d******************************.\n', k);
% 
eta = params.eta;
epsilon = params.epsilon;
D_update_method = params.D_update_method;
new_elements = params.new_elements;
lambda_D = params.lambda_D;
mu = params.mu;
data_type = params.data_type;
noise = params.noise;
True_nonzero_frac = params.True_nonzero_frac;
nonzero_frac = params.nonzero_frac;
test_or_train = params.test_or_train;
%
% 
% Important!
% sahil comments start.
% this is a bug. number of nonzero is decided based upon number of
% dictionary elements. This doesn't have anything to do with dimension of
% input data or the corresponding dimension of dictionary elements.
% sahil: since the number of dictionary elements in the algorithms, the
% number of nonzero should be decided not here but just before learning C.
% sahil end.
% sahil commenting the the line below (instead passing nonzero_frac to the
% appropriate functions).
% nonzero_C = floor(n*nonzero_frac);  % size of compressed representation - fraction of nonzeros as compared to input dim
% % columns of data: input signals (samples)

%%%%%%%%%%%%%%%%%%% simulated datasets %%%%%%%%%%%%%%%%%%%

% generate two different datasets this way, using different dictionaries

%%% random data generator  %%%%%%%%
% nonzero_C = floor(n*nonzero_frac);
% 
%  true_dim = floor(n/2);  %k;
%  [D0_true,C0_true,data0] = simulate(data_type,n,2*true_dim,T+T,nonzero_C,noise);
%  test_data0 = data0(:,T+1:end);
%  data0 = data0(:,1:T);
%  [D_true,C_true,data] = simulate(data_type,n,true_dim,T+T,nonzero_C,noise); % use half for training, half for testing
%  train_data = data(:,1:T);
%  test_data = data(:,T+1:end);
 
 %%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% real images %%%%%%%%%%%%%%%%%%%%%%
is_patches = false;
% 
if is_patches
     %real images (patches)
    [data0, test_data0] = boat_patches(T);
    [train_data, test_data, n] = lena_patches(T);
else
    % or real images itself (Sahil)
    [train_data_map, test_data_map, n] = cifar_images_online(true, T);
    % sea   
    train_data = train_data_map{72};
    assert (size(train_data, 2) == T);
    test_data = test_data_map{72};
    assert (size(test_data, 2) == T);
    % each column is a data point.
    data0 = train_data_map{89};
    assert (size(data0, 2) == T);
    test_data0 = test_data_map{89};
    assert (size(test_data0, 2) == T);
end
%%%%%%%%%%%%%%%%%%%%%%%%% real images %%%%%%%%%%%%%%%%%%%%%%


% learn a dictionary using ``standard way'' on the first dataset
% this would correspond to an 'adult' brain (dictionary <=> link weights)

% sahil: initializing the dictionary here, and normalizing (to see what norm and which dimension, I guess first dimension)
D_init = normalize(rand(n,k));
% 
% sahil: as mentioned above, this is a bug. just passing the fraction instead of the actual number of nonzeros in C.
% sahil: commenting the code line below.
% nonzero_C = floor(n*nonzero_frac);  % size of compressed representation - fraction of nonzeros as compared to input dim. 

% fixed-size dictionary (baseline method-SG and Mairal)  

%legend('random-D','neurogenesis','fixed-size-SG','fixed-size-Mairal','location','SouthEast'); 

% random-D: just use the D_init 
% sahil: D_update_method value doesn't seem to be random in the scripts
% main_batch and main_online.
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for random case.\n');
[D0,err00,correl00] = DL(train_data,D_init,nonzero_frac,0,mu,eta,epsilon,T,-1,data_type,D_update_method);

%plot_online_err(params,err00,correl00,err11,correl11,err22,correl22,err33,correl33,err44,correl44,err55,correl55);

%neurogenesis - with GroupMairal 
%[D1,err11,correl11] =  DL_neurogen(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for neurogenesis with Group Mairal.\n');
[D1,err11,correl11] =  DL(train_data,D_init,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');
 

% neurogenesis - with SG
%[D2,err22,correl22] = DL_neurogen(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for neurogenesis with SG.\n');
[D2,err22,correl22] = DL(train_data,D_init,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');


% group-sparse coding (Bengio et al 2009)
% sahil: no new dictionary elements but deleting existing elements
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for Group Mairal.\n');
[D3,err33,correl33] = DL(train_data,D_init,nonzero_frac,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal');

% sahil: discuss this also with Dr. Rish.
%%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly

% SAHIL: lambda_D is zero in this case. so, this is not really sparse. (DISCUSS WITH DR. RISH.)
%% fixed-size-SG
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for SG (no sparsity though).\n');
[D4,err44,correl44] = DL(train_data,D_init,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'SG');  


% fixed-size-Mairal
fprintf('\n\n\n....................................')
fprintf('Learning the dictionary model for Mairal.\n');
[D5,err55,correl55] = DL(train_data,D_init,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'Mairal');

%%%   legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast'); 

% sahil updated the code (the last parameter) for adding suffix to plot names.
fprintf('Generating plots for training error.\n');
plot_online_err(params,err00,correl00,err11,correl11,err22,correl22,err33,correl33,err44,correl44,err55,correl55, 'train');
%  
% 
% 
% 
% sahil added the code below for evaluating the model on test data.
fprintf('Computing test error by computing sparse codings function ...\n');
tic;
[~,err00_test, correl00_test] = sparse_coding(test_data,D0,nonzero_frac,data_type); % random-D
[~,err11_test, correl11_test] = sparse_coding(test_data,D1,nonzero_frac,data_type); % neurogen-group-Mairal
[~,err22_test, correl22_test] = sparse_coding(test_data,D2,nonzero_frac,data_type); % neurogen-SG
[~,err33_test, correl33_test] = sparse_coding(test_data,D3,nonzero_frac,data_type); % groupMairal 
[~,err44_test, correl44_test] = sparse_coding(test_data,D4,nonzero_frac,data_type); % SG       
[~,err55_test, correl55_test] = sparse_coding(test_data,D5,nonzero_frac,data_type); % Mairal 
fprintf('Time to compute was %f.\n', toc);
% sahil also added code line below to generate plots for evaluation on test
% data. (note: the evaluation below this code line is for the other test
% data which is non-stationary w.r.t. these data sets).
fprintf('Generating plots for testing error.\n');
plot_online_err(params,err00_test,correl00_test,err11_test,correl11_test,err22_test,correl22_test,err33_test,correl33_test,err44_test,correl44_test,err55_test,correl55_test, 'test'); 
% 
% 
% 
% 
% 
% 
% 
% %----------  Mairal's dictionary learning
% 
% param.D = D_init;  
% param.lambda  = 1;  
% %           param.iter (number of iterations).  
% param.batchsize = 1; % (optional, size of the minibatch, by default  512)
% param.modeParam=0; % the optimization uses the parameter free strategy of the ICML paper
% [D3 model]=mexTrainDL(data0,param); 
% %-------------
% [C,err3,correl3] = sparse_coding(data0,D3,nonzero_C,data_type);
% learned_k3 = size(D3,2);          

%sahil: todo- in case of non-stationary evaluation, it would be interesting to evaluate without the training. 
% 
% display(test_or_train);
% 
switch test_or_train 
case 'train'
    [C,err0,correl0] = sparse_coding(train_data,D0,nonzero_frac,data_type); % random-D
    [C,err1,correl1] = sparse_coding(train_data,D1,nonzero_frac,data_type);% neurogen-group-Mairal
    [C,err2,correl2] = sparse_coding(train_data,D2,nonzero_frac,data_type); % neurogen-SG
    [C,err3,correl3] = sparse_coding(train_data,D3,nonzero_frac,data_type);    % groupMairal 
    [C,err4,correl4] = sparse_coding(train_data,D4,nonzero_frac,data_type);    % SG       
    [C,err5,correl5] = sparse_coding(train_data,D5,nonzero_frac,data_type);    % Mairal 

case 'test'
    [C,err0,correl0] = sparse_coding(test_data,D0,nonzero_frac,data_type); % random-D
    [C,err1,correl1] = sparse_coding(test_data,D1,nonzero_frac,data_type);% neurogen-group-Mairal
    [C,err2,correl2] = sparse_coding(test_data,D2,nonzero_frac,data_type); % neurogen-SG
    [C,err3,correl3] = sparse_coding(test_data,D3,nonzero_frac,data_type);    % groupMairal 
    [C,err4,correl4] = sparse_coding(test_data,D4,nonzero_frac,data_type);    % SG       
    [C,err5,correl5] = sparse_coding(test_data,D5,nonzero_frac,data_type);    % Mairal 

case 'nonstat'       
    is_update_dictionary_fr_nonstationary = true;
%     
    if is_update_dictionary_fr_nonstationary
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for Random.\n');
         [D0,err0,correl0] = DL(data0,D0,nonzero_frac,0,mu,eta,epsilon,T,-1,data_type,D_update_method); %random-D
        %         
        %     [D1,err1,correl1] = DL_neurogen(data0,D1,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');   %neurogenesis
        %     [D2,err2,correl2] = DL_neurogen(data0,D2,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');% neurogen-SG
        %         
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for neurogen with Group Mairal.\n');
        [D1,err1,correl1] = DL(data0,D1,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');   %neurogenesis
        %         
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for neurogen with SG.\n');
        [D2,err2,correl2] = DL(data0,D2,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');% neurogen-SG
        %         
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for group Mairal.\n');
        [D3,err3,correl3] = DL(data0,D3,nonzero_frac,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal'); %group Mairal
        %         
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for SG (no sparsity though).\n');
        [D4,err4,correl4] = DL(data0,D4,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'SG');  % just SG
        %         
        fprintf('\n\n\n....................................')
        fprintf('Learning the dictionary model for Mairal.\n');
        [D5,err5,correl5] = DL(data0,D5,nonzero_frac,0,mu,eta,epsilon,T,0,data_type,'Mairal');  %fixed-size Mairal
    end
% 
    [C,err0,correl0] = sparse_coding(test_data0,D0,nonzero_frac,data_type); % random-D
    [C,err1,correl1] = sparse_coding(test_data0,D1,nonzero_frac,data_type);% neurogen-group-Mairal
    [C,err2,correl2] = sparse_coding(test_data0,D2,nonzero_frac,data_type); % neurogen-SG
    [C,err3,correl3] = sparse_coding(test_data0,D3,nonzero_frac,data_type);    % groupMairal 
    [C,err4,correl4] = sparse_coding(test_data0,D4,nonzero_frac,data_type);    % SG       
    [C,err5,correl5] = sparse_coding(test_data0,D5,nonzero_frac,data_type);    % Mairal    
end   
% 
% 
% sahil add the two code lines below for generating plots on the
% non-stationary test data evaluation. (note: these non-stationary
% evaluation error measures are also pass out of the function for generating the final plots).
assert(strcmp(test_or_train, 'nonstat'));
fprintf('Generating plots for non-stationary testing error.\n');
plot_online_err(params,err0,correl0,err1,correl1,err2,correl2,err3,correl3,err4,correl4,err5,correl5, 'nonstationarytest');
% 
% 
% 
learned_k0 = size(D0,2);
learned_k1 = size(D1,2);
learned_k2 = size(D2,2);
learned_k3 = size(D3,2);
learned_k4 = size(D4,2);
learned_k5 = size(D5,2);

batch_size = 1; 
max_iter = T/batch_size;
% adult brain: link weights (dictionary) D1; 
% sparse code C1 for previosly seen inputs


% now, provide a new dataset in online fashion, and try to adapt the
% dictionary using (1) standard approach like mairal's (no neurogenesis)
% versus (2) NIDL (neurogenesis-inspired dictionary learning)
 
% evaluation: 
% give the method a batch of samples of some fixed size (maybe even one sample)
% test prediction accuracy on a set-aside batch of fixed size



% i=1;
% for iter = 1:max_iter
% 	next_batch = train_data(:,i:i+batch_size-1);
% 	[D,C,DL_train_err(iter)] = DL(next_batch,D0,nonzeros_C,lambda_D,eta,batch_size);
% 	
% 	[DL_test_code,DL_test_err(iter),DL_test_corr(iter)] = sparse_code(test_data,D_N,lambda_C);
% 
%       [D_N,C_N,NIDL_train_err(iter)] = NIDL(next_batch,D0,nonzeros_C,lambda_D,eta,batch_size);
% 	[NIDL_test_code,NIDL_test_err(iter),NIDL_test_corr(iter)] = sparse_code(test_data,D_N,nonzeros_C);
% 
% 	i = i + batch_size;
% end

% figure(1);
% plot([1:max_iter], DL_test_err, 'k--', [1:max_iter], NIDL_test_err, 'bx-');
