%TO DO:

% 1. objective in SG is different from Mairal (whole history) -
% try ISTA on whole dataset?
%
% 2. implement group-sparse coord descent (Group sparse coding)
%
% 3. test on'unknown' ground-truth dictionary size, with a range of D sizes



% clear all;
% close all;

% addpath '/gsa/yktgsa-p5/01/imaging/MlabTools/ElasticNet';
addpath './ElasticNet';

%n = 64; % number of inputs (simulated or real 8x8 patches)
n=256;% 16x16 patches
Ti=0;

% dictionary learning approaches: D_update_method parameter:
%  'SG' - our stochastic gradient

%These parameters showed good results:
params = struct('n',n,'k',0,'T',0,'eta',0.1,'epsilon',1e-2,'D_update_method','Mairal','new_elements', 0, ...
               'lambda_D',0.03,'mu',0,'data_type','Gaussian', 'noise',5,'True_nonzero_frac',0.2,'nonzero_frac',0.2, ...
               'test_or_train','train','dataname','lena');

% params = struct('n',n,'k',0,'T',0,'eta',0.1,'epsilon',1e-2,'D_update_method','SG','new_elements', -1, ...
%                'lambda_D',-1,'mu',0,'data_type',...
%              'Gaussian', ..., %  'Bernoulli',... 
%            'noise',0.1,'True_nonzero_frac',0.25,'nonzero_frac',0.25, ...
%                'test_or_train','train');

% params.eta = 1;  % learning rate
% params.epsilon = 1e-3; % convergence criterion
% D_update_method = 'Mairal';


% params.D_update_method = 'SG';
% 
% params.new_elements = n/2; % 'birth rate' - the number of new 'neurons'/dict. elements added for each sample
% params.lambda_D = 3.5; % l1/l2 regularization weight controlling killing rate
% 
% params.data_type='Gaussian'; % data_type = 'Bernoulli';
% params.noise = 1; % noise level - stdev of Gaussian distribution  
% params.True_nonzero_frac = 0.2; % fraction of nonzeros AS COMPARED TO THE INPUT DIMENSION
% params.nonzero_frac = 0.2;  % fraction of nonzeros in sparse coding (SC)

adapt='basic'; %'adapt';
for tt=3:3
    switch tt
        case 1
            params.test_or_train = 'train';
        case 2
            params.test_or_train = 'test';
        case 3
            params.test_or_train = 'nonstat';
        otherwise
    end
    
for T =  100 %300]
    Ti = Ti + 1;
    k_array = [25 50 100 150 ]; %[ 25 50  100 150];%(n/2):(n/2):(4*n);
    ki = 0; err = []; correl = [];
    
%     clear err0 correl0 learned_k0;clear err1 correl1 learned_k1;clear err2 correl2 learned_k2;
%     clear err3 correl3 learned_k3;clear err4 correl4 learned_k4;clear err5 correl5 learned_k5;
   
    for k =  k_array; %(end)  % temporary
        ki = ki+1;
        params.n = n; 
        params.k = k; 
        params.T = T; 
        params.new_elements =  10; %floor(k/5); % floor(2*n*log10(k));  % the number of new elements added for each element;
        %params.lambda_D = 1; % 'killing' weight parameter in l1/l2 regularization (need theory on how to set it asymptotically)
   %     [new_elements,err0(ki,:),correl0(ki,:),learned_k0(ki), err1(ki,:),correl1(ki,:),learned_k1(ki), err2(ki,:),correl2(ki,:),learned_k2(ki),err3(ki,:), correl3(ki,:), learned_k3(ki)] = run_experiments(n,k,T);       
        
         [new_elements,err0(ki,:),correl0,learned_k0(ki), ... 
                      err1(ki,:),correl1,learned_k1(ki), ...
                      err2(ki,:),correl2,learned_k2(ki), ...
                      err3(ki,:),correl3,learned_k3(ki), ... 
                      err4(ki,:),correl4,learned_k4(ki), ...
                      err5(ki,:),correl5,learned_k5(ki)] = run_experiments(params);    
   
        correl0_S(ki,:) = correl0(1,:);correl1_S(ki,:) = correl1(1,:);
        correl2_S(ki,:) = correl2(1,:);correl3_S(ki,:) = correl3(1,:);
        correl4_S(ki,:) = correl4(1,:);correl5_S(ki,:) = correl5(1,:);
        
        correl0_P(ki,:) = correl0(2,:);correl1_P(ki,:) = correl1(2,:);
        correl2_P(ki,:) = correl2(2,:);correl3_P(ki,:) = correl3(2,:);
        correl4_P(ki,:) = correl4(1,:);correl5_P(ki,:) = correl5(1,:);

    end
%     figure(tt); 
%     errorbar(k_array,mean(correl0_P'),std(correl0_P'),'k--'); 
%     hold on;        
%     errorbar(k_array,mean(correl1_P'),std(correl1_P'),'bx-'); 
%     errorbar(k_array,mean(correl2_P'),std(correl2_P'),'ro-');  
%     errorbar(k_array,mean(correl3_P'),std(correl3_P'),'gv-'); 
%     
%     legend('random-D','neurogenesis','fixed-size-SG','fixed-size-Mairal','location','SouthEast'); 
%     ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
%     xlabel('dictionary size k');
%     ylabel('Pearson correlation (true, predicted)');
%     ylim([0,1]);    
%      saveas(gcf,sprintf('Figures/%s_corr_n%d_T%d_new%d',params.test_or_train,params.n,params.T,params.new_elements),'fig');
%      %saveas(gcf,sprintf('Figures/%s_corr_n%d_T%d_new%d',params.test_or_train,params.n,params.T,params.new_elements),'eps');
 
    
%     figure(10000+tt); 
%     errorbar(k_array,mean(err0'),std(err0'),'k--'); 
%     hold on;
%     errorbar(k_array,mean(err1'),std(err1'),'bx-'); 
%      errorbar(k_array,mean(err2'),std(err2'),'ro-');
%     errorbar(k_array,mean(err3'),std(err3'),'gv-');
% legend('random-D','neurogenesis','fixed-size-SG','fixed-size-Mairal','location','SouthEast'); 
%     ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
%     xlabel('dictionary size k');
%     ylabel('sum-squared error ');   
%      saveas(gcf,sprintf('Figures/%s_err_n%d_T%d_new%d',params.test_or_train,params.n,params.T,params.new_elements),'fig');
% %      saveas(gcf,sprintf('Figures/%s_err_n%d_T%d_new%d',params.test_or_train,params.n,params.T,params.new_elements),'eps');
% %      saveas(gcf,sprintf('Figures/%s_err_n%d_T%d_new%d',params.test_or_train,params.n,params.T,params.new_elements),'bmp');

    
    figure(1000+tt); hold on;
    plot(k_array,learned_k0,'k--',k_array,learned_k1,'bx-',k_array,learned_k2,'bo-',k_array,learned_k3,'rs-',...
        k_array,learned_k4,'gv-',k_array,learned_k5,'md-');    
    
     legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast'); 
    
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    xlabel('initial dictionary size k');
    ylabel('learned dictionary size');

    saveas(gcf,sprintf('Figures/%s_%s_learn_k_n%d_nz%d_lam%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,100*params.lambda_D,params.T,params.new_elements,adapt),'fig');
    saveas(gcf,sprintf('Figures/%s_%s_learn_k_n%d_nz%d_lam%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,100*params.lambda_D,params.T,params.new_elements,adapt),'png');
 
     
    %%%% plot actual dictionary size vs error or vs correlation
    figure(tt+10); 
    errorbar(learned_k0,mean(correl0_P'),std(correl0_P'),'k--'); 
    hold on;        
    errorbar(learned_k1,mean(correl1_P'),std(correl1_P'),'bx-'); 
    errorbar(learned_k2,mean(correl2_P'),std(correl2_P'),'bo-');  
    errorbar(learned_k3,mean(correl3_P'),std(correl3_P'),'rs-'); 
    errorbar(learned_k4,mean(correl4_P'),std(correl4_P'),'gv-');
    errorbar(learned_k5,mean(correl5_P'),std(correl5_P'),'md-');
    
    legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast');
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    
    
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);    
     saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_lam%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,100*params.lambda_D,params.T,params.new_elements,adapt),'fig');
     saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_lam%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,100*params.lambda_D,params.T,params.new_elements,adapt),'png');
     
     
     figure(10010+tt); 
    errorbar(learned_k0,mean(err0'),std(err0'),'k--'); 
    hold on;
    errorbar(learned_k1,mean(err1'),std(err1'),'bx-'); 
     errorbar(learned_k2,mean(err2'),std(err2'),'bo-');
    errorbar(learned_k3,mean(err3'),std(err3'),'rs-');
        errorbar(learned_k4,mean(err4'),std(err4'),'gv-');
    errorbar(learned_k5,mean(err5'),std(err5'),'md-');
    
  legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast');
 
 ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    xlabel('final dictionary size k');
    ylabel('sum-squared error ');   
     saveas(gcf,sprintf('Figures/%s_%s_err_n%d_nz%d_lamD%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,100*params.lambda_D,params.T,params.new_elements,adapt),'fig');
      saveas(gcf,sprintf('Figures/%s_%s_err_n%d_nz%d_lamD%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,100*params.lambda_D,params.T,params.new_elements,adapt),'png');

     
%%
 
end

end
