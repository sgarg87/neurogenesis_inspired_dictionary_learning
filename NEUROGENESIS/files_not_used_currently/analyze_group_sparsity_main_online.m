clear;
close all;
% 
addpath './ElasticNet';
% 
n=1024; %32x32 %256;% 16x16 patches
Ti=0;

%These parameters showed good results:
params = struct('n',n,'k',0,'T',0,'eta',0.1,'epsilon',1e-2,'D_update_method','Mairal','new_elements', 0, ...
               'lambda_D',0.03,'mu',0,'data_type','Gaussian', 'noise',5,'True_nonzero_frac',0.2,'nonzero_frac',1.0, ...
               'test_or_train','train','dataname','cifar100');

params.adapt='basic';
tt = 3;
params.test_or_train = 'nonstat';
    
for T = 500
    Ti = Ti + 1;
    k_array = 5:20:205; %[25 50 100 150]; %(n/2):(n/2):(4*n);
    ki = 0; err = []; correl = [];
    %        
    for k =  k_array; %(end)  % temporary
        ki = ki+1;
        params.n = n; 
        params.k = k; 
        params.T = T;
        %         
        [ err0(ki,:),correl0,learned_k0(ki), ... 
                      err1(ki,:),correl1,learned_k1(ki) ] = analyze_group_sparsity_run_experiments(params);    
   
        correl0_S(ki,:) = correl0(1,:);correl1_S(ki,:) = correl1(1,:);
        correl0_P(ki,:) = correl0(2,:);correl1_P(ki,:) = correl1(2,:);
    end
    
    figure(1000+tt); hold on;
    plot(k_array,learned_k0,'ro',k_array,learned_k1,'bx');    
    %     
    legend('groupMairal', 'Mairal','location','SouthEast');
    %     
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    xlabel('initial dictionary size k');
    ylabel('learned dictionary size');
    % 
    saveas(gcf,sprintf('Figures_group/%s_%s_learn_k_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
    saveas(gcf,sprintf('Figures_group/%s_%s_learn_k_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    %sahil added code closing the figure after the saving.
    close(gcf);
     
    %%%% plot actual dictionary size vs error or vs correlation
    figure(tt+10000); 
    errorbar(learned_k0,mean(correl0_P'),std(correl0_P'),'ro');
    hold on;        
    errorbar(learned_k1,mean(correl1_P'),std(correl1_P'),'bx'); 
    
    legend('groupMairal','Mairal','location','SouthEast');
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    
    
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);
     saveas(gcf,sprintf('Figures_group/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
     saveas(gcf,sprintf('Figures_group/%s_%s_corr_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    %sahil added code closing the figure after the saving.    
     close(gcf);     
     
     %%err
         %%%% plot actual dictionary size vs error or vs correlation
    figure(tt+100);
    errorbar(learned_k0,mean(err0'),std(err0'),'ro');
    hold on;        
    errorbar(learned_k1,mean(err1'),std(err1'),'bx');
    
    legend('groupMairal','Mairal','location','SouthEast');
    ss = sprintf('%s: input dim n=%d, samples = %d',params.test_or_train,n,T); title(ss);
    
    
    xlabel('final dictionary size k');
    ylabel('MSE');
    ylim([0,1]);    
     saveas(gcf,sprintf('Figures_group/%s_%s_err_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'fig');
     saveas(gcf,sprintf('Figures_group/%s_%s_err_n%d_nz%d_T%d_new%d%s',params.dataname,params.test_or_train,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt),'png');
    %sahil added code closing the figure after the saving.    
     close(gcf);     
 
%%
end
