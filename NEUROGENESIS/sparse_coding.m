function [C,err,correl] = sparse_coding(x,D,nonzero_frac, data_type)

% learn sparse code C for a given data and a given dictionary D

% parameters:
% nonzeros_C - regularization parameter for optimizing sparse code C: for LARS, # of nonzeros
%              if nonzeros = -1, the sparsity level is selected automatically as  best-fitting for each given
% data_type - 'Gaussian', or 'Bernoulli', or another exp-family

%method used: LARS-EN
% 
% sahil: not fully understood the purpose of this, I guess this is in regards to
% the extension of the LASSO based sparsity learning Marial et al to
% Elastic Net. nonzeros_C represents cofficient for the L1 norm which
% controls the sparsity and lambda2_C is for the group sparsity using L2
% norm.
lambda2_C = 0.00001; %0;  % LASSO
% 
k = size(D,2);
% 
C = [];
% 
%
% sahil added code to compute number of nonzeros.
% nonzeros_C = floor(k*nonzero_frac);
% display(nonzeros_C);
nonzeros_C = floor(size(D, 1)*nonzero_frac);
% 
% 
% sahil: iterating over all the data points in the batch (x)
% sahil: each column in x is a single data point
for i = 1:size(x,2)

	if nonzeros_C >= 0  % specified sparsity level
    % sahil: y is current data to be processed.    
	   D1 = D; y = x(:,i);
    % sahil: inefficiency here ?
       D1 = normalize(D);
    % sahil: inefficiency here ?
        y = center(x(:,i));
    % sahil: I guess these computed betas are alphas in the terminology of the Mairal et al
        betas = larsen(D1, y, lambda2_C, -nonzeros_C, 0);
    % sahil: I guess there is only row in betas corresponding to the beta learned for y.       
		sol = betas(end,:)';
        if nnz(isnan(sol))
            display 'NaN in sol'
            pause;
        end
       res = y-D1*sol; 
       err(i) = res'*res;
	   nonzeros(i) = nonzeros_C;

	else % if no sparsity level is specified (nonzeros_C = -1), choose best-fitting sparsity for a given sample
		betas = larsen(D, x(:,i), lambda2_C, -k, 0);
        % sahil: I guess, this code is written differently from the above if block
        % but correct anyways.
        % sahil: default code is for sparsity. if use this code, make sure it is correct code.
		all_res = repmat(x(:,i),1,size(betas,1)) - predicted;
		all_err = diag(all_res'*all_res);
        % sahil: are there multiple solutions of beta for same data ?        
		[err(i),nonzeros(i)] = min(all_err);
		sol = betas(nonzeros(i),:)';
	end 		
	correl_Spearman(i)  = corr(x(:,i),D1*sol,'type','Spearman');   
    correl_Pearson(i) = corr(x(:,i),D1*sol,'type','Pearson');
   
	C = [C sol];
    if nnz(isnan(C))
            display 'NaN in code'
            pause;
    end

    
end
correl = [correl_Spearman; correl_Pearson];

