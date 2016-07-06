function [C,err,correl] = sparse_coding(x,D, params)
    % learn sparse code C for a given data and a given dictionary D using LARS-EN.
    %
    %
    if params.is_sparse_computations
        if ~issparse(D)
            D = sparse(D);
        end
    end
    %     
    nonzero_frac = params.nonzero_frac;
    lambda2_C = params.lambda2_C;
    %
    k = size(D,2);
    C = [];
    nonzeros_C = floor(size(D, 1)*nonzero_frac);
    %
    for i = 1:size(x,2)
        if nonzeros_C >= 0  % specified sparsity level
            %% how important is to normalize the dictionary before learning the sparse codings and also the center of x.
            %% not sure if the normalization and the centering being done along correct dimensions.
            D1 = normalize(D);
            y = center(x(:,i));
            %
            if strcmp(params.coding_sparse_algo, 'lars')
                betas = larsen(D1, y, lambda2_C, -nonzeros_C, 0);
                sol = betas(end,:)';
            elseif strcmp(params.coding_sparse_algo, 'proximal')
                [sol, ~] = lsqr(D1, y);
                if nonzeros_C < k
                    sparse_coding_lam = binary_search_proximal_threshold(sol, nonzeros_C, max(0.01*nonzeros_C, 1));
                    sol = sign(sol).*max(abs(sol)-sparse_coding_lam, 0);
%                     fprintf('\nsc-nnz: %d', nnz(sol));
                end
            else
                error('no such method for codings sparsity.')
            end
            assert(~nnz(isnan(sol)));
            assert(nnz(sol) ~= 0);
            %             
            res = y-D1*sol;
            err(i) = res'*res;
            nonzeros(i) = nonzeros_C;
        else
            error('Sahil: this code block is not executed currently, and does not seem right. So, also the recent changes on proximal based sparsity implemented only for the above block and not this one.')
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
    %     
    correl = [correl_Spearman; correl_Pearson];
    assert(~nnz(isnan(correl)));
    %
    if params.is_sparse_computations
        if ~issparse(C)
            C = sparse(C);
        end
    end
end

