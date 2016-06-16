function [C,err,correl] = sparse_coding(x,D,nonzero_frac)
    % learn sparse code C for a given data and a given dictionary D using LARS-EN.
    lambda2_C = 0.00001; %0;  % LASSO
    k = size(D,2);
    C = [];
    nonzeros_C = floor(size(D, 1)*nonzero_frac);
    % 
    for i = 1:size(x,2)
        if nonzeros_C >= 0  % specified sparsity level
            D1 = normalize(D);
            y = center(x(:,i));
            betas = larsen(D1, y, lambda2_C, -nonzeros_C, 0);
            %         
            sol = betas(end,:)';
            assert(~nnz(isnan(sol)));
            res = y-D1*sol; 
            %         
            err(i) = res'*res;
            nonzeros(i) = nonzeros_C;
        else
            % if no sparsity level is specified (nonzeros_C = -1), choose best-fitting sparsity for a given sample
            betas = larsen(D, x(:,i), lambda2_C, -k, 0);
            all_res = repmat(x(:,i),1,size(betas,1)) - predicted;
            all_err = diag(all_res'*all_res);
            [err(i),nonzeros(i)] = min(all_err);
            sol = betas(nonzeros(i),:)';
        end 		
        %     
        correl_Spearman(i)  = corr(x(:,i),D1*sol,'type','Spearman');   
        correl_Pearson(i) = corr(x(:,i),D1*sol,'type','Pearson');
        %    
        C = [C sol];
        assert(~nnz(isnan(C)));
    end
    correl = [correl_Spearman; correl_Pearson];
end
