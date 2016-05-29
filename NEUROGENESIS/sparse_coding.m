function [C,err,correl] = sparse_coding(x,D,nonzeros_C,data_type)

% learn sparse code C for a given data and a given dictionary D

% parameters:
% nonzeros_C - regularization parameter for optimizing sparse code C: for LARS, # of nonzeros
%              if nonzeros = -1, the sparsity level is selected automatically as  best-fitting for each given 
% data_type - 'Gaussian', or 'Bernoulli', or another exp-family

%method used: LARS-EN

lambda2_C = 0.00001; %0;  % LASSO
k = size(D,2);
C = [];

for i = 1:size(x,2)

	if nonzeros_C >= 0  % specified sparsity level
	   D1 = D; y = x(:,i);
       D1 = normalize(D);
        y = center(x(:,i));
        betas = larsen(D1, y, lambda2_C, -nonzeros_C, 0);
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
		predicted = D*betas';
		all_res = repmat(x(:,i),1,size(betas,1)) - predicted;
		all_err = diag(all_res'*all_res);
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

