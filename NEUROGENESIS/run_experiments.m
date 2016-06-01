function [new_elements,err0, correl0, learned_k0, err1, correl1, learned_k1, ...
    err2, correl2,learned_k2,err3, correl3, learned_k3, ...
    err4, correl4, learned_k4, err5, correl5, learned_k5] = run_experiments(params)
% 
% what is n, k, t
n = params.n;
k = params.k;
T = params.T;
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
nonzero_C = floor(n*nonzero_frac);  % size of compressed representation - fraction of nonzeros as compared to input dim
 
% columns of data: input signals (samples)

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
if is_patches
     %real images (patches)
    [data0, test_data0] = boat_patches(T);
    [train_data, test_data, n] = lena_patches(T);
else
    % or real images itself (Sahil)
    [train_data, test_data, data0, test_data0, n] = cifar_images();
end
%%%%%%%%%%%%%%%%%%%%%%%%% real images %%%%%%%%%%%%%%%%%%%%%%


% learn a dictionary using ``standard way'' on the first dataset
% this would correspond to an 'adult' brain (dictionary <=> link weights)

% sahil: initializing the dictionary here, and normalizing (to see what norm and which dimension, I guess first dimension)
D_init = normalize(rand(n,k));
% sahil: not sure if dictionary vectors are sparse here or or the number of vectors sparse in general
nonzero_C = floor(n*nonzero_frac);  % size of compressed representation - fraction of nonzeros as compared to input dim
 

% fixed-size dictionary (baseline method-SG and Mairal)  

%legend('random-D','neurogenesis','fixed-size-SG','fixed-size-Mairal','location','SouthEast'); 

% random-D: just use the D_init 
% sahil: D_update_method value doesn't seem to be random in the scripts
% main_batch and main_online.
[D0,err00,correl00] = DL(train_data,D_init,nonzero_C,0,mu,eta,epsilon,T,-1,data_type,D_update_method);

%plot_online_err(params,err00,correl00,err11,correl11,err22,correl22,err33,correl33,err44,correl44,err55,correl55);

%neurogenesis - with GroupMairal 
%[D1,err11,correl11] =  DL_neurogen(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');
[D1,err11,correl11] =  DL(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');
 

% neurogenesis - with SG
%[D2,err22,correl22] = DL_neurogen(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');
[D2,err22,correl22] = DL(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');


% group-sparse coding (Bengio et al 2009)
% sahil: no new dictionary elements but deleting existing elements
[D3,err33,correl33] = DL(train_data,D_init,nonzero_C,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal');

%%  TO DEBUG: group Mairal with lambda_D = 0 does not seem to work properly

%% fixed-size-SG
[D4,err44,correl44] = DL(train_data,D_init,nonzero_C,0,mu,eta,epsilon,T,0,data_type,'SG');  


% fixed-size-Mairal
[D5,err55,correl55] = DL(train_data,D_init,nonzero_C,0,mu,eta,epsilon,T,0,data_type,'Mairal');

%%%   legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast'); 

 plot_online_err(params,err00,correl00,err11,correl11,err22,correl22,err33,correl33,err44,correl44,err55,correl55);

 

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

           
 switch test_or_train 
     case 'train'
        [C,err0,correl0] = sparse_coding(train_data,D0,nonzero_C,data_type); % random-D
        [C,err1,correl1] = sparse_coding(train_data,D1,nonzero_C,data_type);% neurogen-group-Mairal
        [C,err2,correl2] = sparse_coding(train_data,D2,nonzero_C,data_type); % neurogen-SG
        [C,err3,correl3] = sparse_coding(train_data,D3,nonzero_C,data_type);    % groupMairal 
        [C,err4,correl4] = sparse_coding(train_data,D4,nonzero_C,data_type);    % SG       
        [C,err5,correl5] = sparse_coding(train_data,D5,nonzero_C,data_type);    % Mairal 
            
     case 'test'
        [C,err0,correl0] = sparse_coding(test_data,D0,nonzero_C,data_type); % random-D
        [C,err1,correl1] = sparse_coding(test_data,D1,nonzero_C,data_type);% neurogen-group-Mairal
        [C,err2,correl2] = sparse_coding(test_data,D2,nonzero_C,data_type); % neurogen-SG
        [C,err3,correl3] = sparse_coding(test_data,D3,nonzero_C,data_type);    % groupMairal 
        [C,err4,correl4] = sparse_coding(test_data,D4,nonzero_C,data_type);    % SG       
        [C,err5,correl5] = sparse_coding(test_data,D5,nonzero_C,data_type);    % Mairal 
        
     
    case 'nonstat'       
     [D0,err0,correl0] = DL(data0,D0,nonzero_C,0,mu,eta,epsilon,T,-1,data_type,D_update_method); %random-D
%     [D1,err1,correl1] = DL_neurogen(data0,D1,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');   %neurogenesis
%     [D2,err2,correl2] = DL_neurogen(data0,D2,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');% neurogen-SG
     [D1,err1,correl1] = DL(data0,D1,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'GroupMairal');   %neurogenesis
     [D2,err2,correl2] = DL(data0,D2,nonzero_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,'SG');% neurogen-SG

     [D3,err3,correl3] = DL(data0,D3,nonzero_C,lambda_D,mu,eta,epsilon,T,0,data_type,'GroupMairal'); %group Mairal
     [D4,err4,correl4] = DL(data0,D4,nonzero_C,0,mu,eta,epsilon,T,0,data_type,'SG');  % just SG
     [D5,err5,correl5] = DL(data0,D5,nonzero_C,0,mu,eta,epsilon,T,0,data_type,'Mairal');  %fixed-size Mairal

        [C,err0,correl0] = sparse_coding(test_data0,D0,nonzero_C,data_type); % random-D
        [C,err1,correl1] = sparse_coding(test_data0,D1,nonzero_C,data_type);% neurogen-group-Mairal
        [C,err2,correl2] = sparse_coding(test_data0,D2,nonzero_C,data_type); % neurogen-SG
        [C,err3,correl3] = sparse_coding(test_data0,D3,nonzero_C,data_type);    % groupMairal 
        [C,err4,correl4] = sparse_coding(test_data0,D4,nonzero_C,data_type);    % SG       
        [C,err5,correl5] = sparse_coding(test_data0,D5,nonzero_C,data_type);    % Mairal 
     
end   
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
