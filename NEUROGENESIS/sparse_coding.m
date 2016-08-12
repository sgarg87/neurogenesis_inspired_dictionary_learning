function [C, err, correl] = sparse_coding(x, D, params)
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
    input_dim = size(D, 1);
    %     
    k = size(D,2);
    C = [];
    nonzeros_C = floor(size(D, 1)*nonzero_frac);
    assert(nonzeros_C > 0);
    %
    for i = 1:size(x,2)
        if nonzeros_C >= 0  % specified sparsity level
            %% how important is to normalize the dictionary before learning the sparse codings and also the center of x.
            %% not sure if the normalization and the centering being done along correct dimensions.
            %% todo: see the effect of centering the data when sparse data. esp. for less sparse data, it may be destorying the sparsity.
            D1 = normalize(D);
%% Sahil commented the code for the centering. Discuss with Irina.
            y = x(:, i);
%             y = center(y);
% 
%             %% sahil: centering of the data can make it a zero vector. In that case, sparse coding becomes zero and correspondingly the correlation metrics are nan.
%             %% to deal with this issue, adding a very small noise in the data.
%             y = y + (min(abs(y))*1e-2)*rand(size(y));
            %             
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
            %
%             if nnz(sol) == 0
%                 pause;
%             end
            %             
            assert(~nnz(isnan(sol)));
%             assert(nnz(sol) ~= 0);
            %             
            % sahil incorporated the change for normalizing with the mean square error with the input dimension           
%             res = (y - D1*sol);
            res = (center(y) - D1*sol);
            err(i) = (res'*res)/input_dim;
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
        % sahil added condition for a zero vector sol.
        if ~nnz(sol)
            correl_Spearman(i)  = 1e-10;
            correl_Pearson(i) = 1e-10;
        else
            correl_Spearman(i)  = corr(x(:,i),D1*sol,'type','Spearman');
            correl_Pearson(i) = corr(x(:,i),D1*sol,'type','Pearson');
            if nnz(isnan(correl_Spearman(i))) || nnz(isnan(correl_Pearson(i))) 
                pause;
            end
            assert(~nnz(isnan(correl_Spearman(i))));
            assert(~nnz(isnan(correl_Pearson(i))));
        end
        %    
        C = [C sol];
        assert(~nnz(isnan(C)));
    end
    %     
    correl = [correl_Spearman; correl_Pearson];
    %
    if nnz(isnan(correl))
        pause;
    end
    assert(~nnz(isnan(correl)));
    %
    if params.is_sparse_computations
        if ~issparse(C)
            C = sparse(C);
        end
    end
end

