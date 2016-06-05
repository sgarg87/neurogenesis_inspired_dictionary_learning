%TO DO (Irina):
% 1. objective in SG is different from Mairal (whole history) -
% try ISTA on whole dataset?
% 2. implement group-sparse coord descent (Group sparse coding)
% 3. test on'unknown' ground-truth dictionary size, with a range of D sizes

clear;
close all;
addpath './ElasticNet';
% 
% dictionary learning approaches: D_update_method parameter:
%  'SG' - our stochastic gradient

%These parameters showed good results:
% params = struct('n',n,'k',0,'T',0,'eta',0.1,'epsilon',1e-2,'D_update_method','Mairal','new_elements', 0, ...
%                'lambda_D',0.03,'mu',0,'data_type','Gaussian', 'noise',5,'True_nonzero_frac',0.2,'nonzero_frac',0.2, ...
%                'test_or_train','train','dataname','lena');
% sahil trying out some changes in the parameters.
params = struct('eta',0.1,'epsilon',1e-2,'D_update_method','Mairal','new_elements', 0, ...
               'lambda_D',0.01,'mu',0,'data_type','Gaussian', 'noise',5,'True_nonzero_frac',0.2,'nonzero_frac',0.2, ...
               'test_or_train','train','dataname','cifar100');
n = 1024;
params.n = n;
T = 20;
params.T = T;
params.test_or_train = 'online';
params.adapt='basic'; %'adapt';
params.new_elements =  10; 
%floor(k/5); % floor(2*n*log10(k));  % the number of new elements added for each element;
%params.lambda_D = 1; % 'killing' weight parameter in l1/l2 regularization (need theory on how to set it asymptotically)
%     [new_elements,err0(ki,:),correl0(ki,:),learned_k0(ki), err1(ki,:),correl1(ki,:),learned_k1(ki), err2(ki,:),correl2(ki,:),learned_k2(ki),err3(ki,:), correl3(ki,:), learned_k3(ki)] = run_experiments(n,k,T);       
%     
%(n/2):(n/2):(4*n);
k_array = [5 10 15 20 25]; % 15 20 25];
% 
ki = 0; err = []; correl = [];
%        
for k =  k_array;
    ki = ki+1;
    %   
    params.k = k; 
    % 
    [new_elements,err0(ki,:),correl0,learned_k0(ki), ... 
                  err1(ki,:),correl1,learned_k1(ki), ...
                  err2(ki,:),correl2,learned_k2(ki), ...
                  err3(ki,:),correl3,learned_k3(ki), ... 
                  err4(ki,:),correl4,learned_k4(ki), ...
                  err5(ki,:),correl5,learned_k5(ki)] = brain_online_eval(params);    
    % 
    correl0_S(ki,:) = correl0(1,:);correl1_S(ki,:) = correl1(1,:);
    correl2_S(ki,:) = correl2(1,:);correl3_S(ki,:) = correl3(1,:);
    correl4_S(ki,:) = correl4(1,:);correl5_S(ki,:) = correl5(1,:);
    % 
    correl0_P(ki,:) = correl0(2,:);correl1_P(ki,:) = correl1(2,:);
    correl2_P(ki,:) = correl2(2,:);correl3_P(ki,:) = correl3(2,:);
    correl4_P(ki,:) = correl4(2,:);correl5_P(ki,:) = correl5(2,:);
end
% 
tt = 3;
figure(1000+tt); hold on;
plot(k_array,learned_k0,'k--',k_array,learned_k1,'bx-',k_array,learned_k2,'bo-',k_array,learned_k3,'rs-',...
k_array,learned_k4,'gv-',k_array,learned_k5,'md-');    

legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast'); 

ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
xlabel('initial dictionary size k');
ylabel('learned dictionary size');
% 
saveas(gcf,sprintf('Figures/%s_%s_learn_k_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
saveas(gcf,sprintf('Figures/%s_%s_learn_k_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
%sahil added code closing the figure after the saving.    
close(gcf);

%%%% plot actual dictionary size vs error or vs correlation
figure(tt+10000); 
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
saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
saveas(gcf,sprintf('Figures/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
%sahil added code closing the figure after the saving.    
close(gcf);     

%%err
 %%%% plot actual dictionary size vs error or vs correlation
figure(tt+100);
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
ylabel('MSE');
ylim([0,1]);    
saveas(gcf,sprintf('Figures/%s_%s_err_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
saveas(gcf,sprintf('Figures/%s_%s_err_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
%sahil added code closing the figure after the saving.    
close(gcf);


