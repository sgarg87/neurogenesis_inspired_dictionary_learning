function plot_online_err(params, err00,correl00,err11,correl11,err22,correl22,err33,correl33,err44,correl44,err55,correl55, suffix)

 %%%% plot actual dictionary size vs error or vs correlation

    batch_size=20;
    T = length(err00);
    
    i1=1;
    for i=1:floor(T/batch_size)
        i2 = i1+batch_size-1;
        m_err00(i) = mean(err00(i1:i2));m_err11(i) = mean(err11(i1:i2)); m_err22(i) = mean(err22(i1:i2)); 
        m_err33(i) = mean(err33(i1:i2));m_err44(i) = mean(err44(i1:i2)); m_err55(i) = mean(err55(i1:i2)); 
        
        s_err00(i) = std(err00(i1:i2));s_err11(i) = std(err11(i1:i2)); s_err22(i) = std(err22(i1:i2)); 
        s_err33(i) = std(err33(i1:i2));s_err44(i) = std(err44(i1:i2)); s_err55(i) = std(err55(i1:i2)); 
        
        i1 = i2+1;
    end
    
    i1=1;
    for i=1:floor(T/batch_size)
        i2 = i1+batch_size-1;
        m_correl00(i) = mean(correl00(2,i1:i2));m_correl11(i) = mean(correl11(2,i1:i2)); m_correl22(i) = mean(correl22(2,i1:i2)); 
        m_correl33(i) = mean(correl33(2,i1:i2));m_correl44(i) = mean(correl44(2,i1:i2)); m_correl55(i) = mean(correl55(2,i1:i2)); 
        
        s_correl00(i) = std(correl00(2,i1:i2));s_correl11(i) = std(correl11(2,i1:i2)); s_correl22(i) = std(correl22(2,i1:i2)); 
        s_correl33(i) = std(correl33(2,i1:i2));s_correl44(i) = std(correl44(2,i1:i2)); s_correl55(i) = std(correl55(2,i1:i2)); 
        
        i1 = i2+1;
    end
    
    %%%%%%%%%%%%
    figure(2000+params.k);
    errorbar(1:floor(T/batch_size),m_err00,s_err00,'k--'); 
    hold on;        
    errorbar(1:floor(T/batch_size),m_err11,s_err11,'bx-'); 
    errorbar(1:floor(T/batch_size),m_err22,s_err22,'bo-');  
    errorbar(1:floor(T/batch_size),m_err33,s_err33,'rs-'); 
    errorbar(1:floor(T/batch_size),m_err44,s_err44,'gv-');
    errorbar(1:floor(T/batch_size),m_err55,s_err55,'md-');
    
    legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast');
    ss = sprintf('Online:  input dim n=%d, samples = %d ',params.n, params.T); title(strcat(ss, suffix));
    
    
    xlabel('iteration (batch)');
    ylabel('MSE (true, predicted)');
    ylim([0,1]);    
    %sahil updated the code for adding the suffix parameter in the figure file names.
    file_path = sprintf('Figures/online_%s_k%d_err_n%d_nz%d_T%d_new%d%s__%s',params.dataname,params.k,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt, suffix);
%     display(file_path);
    saveas(gcf, file_path, 'fig');
    saveas(gcf, file_path, 'png');
    %sahil added code closing the figure after the saving.    
    close(gcf);
%      
    figure(3000+params.k);
    errorbar(1:floor(T/batch_size),m_correl00,s_correl00,'k--'); 
    hold on;        
    errorbar(1:floor(T/batch_size),m_correl11,s_correl11,'bx-'); 
    errorbar(1:floor(T/batch_size),m_correl22,s_correl22,'bo-');  
    errorbar(1:floor(T/batch_size),m_correl33,s_correl33,'rs-'); 
    errorbar(1:floor(T/batch_size),m_correl44,s_correl44,'gv-');
    errorbar(1:floor(T/batch_size),m_correl55,s_correl55,'md-');
    
    legend('random-D','neurogen-groupMairal','neurogen-SG','groupMairal','SG','Mairal','location','SouthEast');
    ss = sprintf('Online:  input dim n=%d, samples = %d',params.n, params.T); title(ss);
    
    
    xlabel('iteration (batch)');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);
    %sahil updated the code for adding the suffix parameter in the figure file names.      
    saveas(gcf,sprintf('Figures/online_%s_k%d_corr_n%d_nz%d_T%d_new%d%s__%s',params.dataname,params.k,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt, suffix),'fig');
    saveas(gcf,sprintf('Figures/online_%s_k%d_corr_n%d_nz%d_T%d_new%d%s__%s',params.dataname,params.k,params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt, suffix),'png');
    %sahil added code closing the figure after the saving.    
    close(gcf);
end