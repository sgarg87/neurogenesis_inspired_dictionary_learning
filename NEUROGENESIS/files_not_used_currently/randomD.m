% generate input samples: image patches

I=double(imread('spams-matlab/data/lena.png'))/255;
% extract 8 x 8 patches
X=im2col(I,[8 8],'sliding');
X=X-repmat(mean(X),[size(X,1) 1]);
X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

% parameters
n = size(X,1);  % dimensionality of input samples
Tmax = size(X,2);  % number of input samples
nonzero_frac = 0.1; % fraction of nonzeros in sparse code

k_array = [25 50 64 75 100]; %[25 50 64 100 200 300];  
ki = 0; err = []; correl = [];

T = 100; data_type='Gaussian';
tt=1;

ww = ['k--';'k-.';'bx-';'ro-';'gv-'];
ii=0;

for nonzero_frac=[ 0.1 0.5 0.7 1]
    ii=ii+1;
%for =[100 500 1000] 
    X_input = X(:,1:T);
   ki = 0; 
  for k =  k_array %(end)  % temporary
        ki = ki+1;
        
        % generate random dictionary with k elements
        D = normalize(rand(n,k)); 
        if nonzero_frac<0
            s = -nonzero_frac;
        else
            s = floor(n*nonzero_frac); % sparsity/compressed representation 
        end
        if ~s  s=1; end

        [C,err(ki,:),correl] = sparse_coding(X_input,D,s,data_type);
        correl_S(ki,:) = correl(1,:); 
        correl_P(ki,:) = correl(2,:);
  end
  
  figure(tt); 
    errorbar(k_array,mean(correl_P'),std(correl_P'),ww(ii)); hold on; 
  figure(100+tt);
    errorbar(k_array,mean(err'),std(err'),ww(ii)); hold on;
    
    clear correl err correl_S correl_P;
end
   tt=1;
   figure(tt); 
    %errorbar(k_array,mean(correl_P'),std(correl_P'),'k--'); hold on;      
    legend('s=1','s=3','nz=0.1','nz=0.5','nz=1','location','SouthEast'); 
    ss = sprintf('Correlation: input dim  n=%d, samples = %d, nz=%.2f',n,T,nonzero_frac); title(ss);
    xlabel('dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);    
     saveas(gcf,sprintf('Figures/rnd_corr_n%d_T%d',n,T),'fig');
     saveas(gcf,sprintf('Figures/rnd_corr_n%d_T%d',n,T),'bmp'); 
     
    
    figure(100+tt); 
  legend('s=1','s=3','nz=0.1','nz=0.5','nz=1','location','SouthEast');
    ss = sprintf('MSE: input dim n=%d, samples = %d, nz=%.2f',n,T,nonzero_frac); title(ss);
    xlabel('dictionary size k');
    ylabel('sum-squared error ');   
    ylim([0,1]);
    saveas(gcf,sprintf('Figures/rnd_mse_n%d_T%d',n,T),'fig');
    saveas(gcf,sprintf('Figures/rnd_mse_n%d_T%d',n,T),'bmp');
    
   
