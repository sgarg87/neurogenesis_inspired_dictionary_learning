function  [D] = updateD(D_old,code,x,params,D_update_method,A,B)
    % update dictionary, given the current dictionary, sparse code, data and parameters
    % D_update_method
    %
    % Sahil added code for sparse computations on dictionary.
    % note: this would be inefficient for first iteration
    % if the initialization of the dictionary is not sparse.
    if ~issparse(D_old) && params.is_sparse_computations
        D_old = sparse(D_old);
    end
    %
    n = size(D_old,1);
    %
    % 
    k = size(D_old,2);
    D = D_old;
    %     
    max_num_iter = min(max(0.01*n, 5), 20);
%     max_num_iter = 200;
    %
    switch D_update_method
        case 'SG' %stochastic gradient with thresholding, i.e. proximal method
            %               
            converged = 0;
            curr_count = 0;
            %
            if params.is_sg_newton
                tic;
                A_inv = inv(A);
                fprintf('Time to inverse was %f.\n', toc);
                clear start_inv_time;
            end
            %             
            while ~converged
                curr_count = curr_count + 1;
                %                 
                if curr_count > max_num_iter
                    break;
                end
                Dprev = D;
                %     
                % 
                if params.is_sg_newton
                    eta = 1;
                else
                    eta = params.eta;
                end
                % 
                %                 
                if nnz(isinf(D))
                    display 'infty in D'
                    pause;
                end
                if nnz(isnan(D))
                    display 'NaNs in D'
                    pause;
                end
                % 
                    for j=1:k
                        %                         
                        % gradient step; first index - input dimension, second - dictionary element
                        % sahil: "link_func(D*code,data_type) - x" is negative of error we make in inferring x.
                        % todo: make this step more efficient by computing the change only for the current block (jth dictionary element)
                        %                         
                        if params.is_sg_memory_based
                            D_change = (D*A) - B;
                            if params.is_sg_newton
                                D_change = D_change*A_inv;
                            end
                        else
                            D_change = (link_func(D*code,params.data_type) - x)*code';
                        end
                        %                         
                        % 
                        u = D(:, j) - eta*D_change(:, j);
                        % 
                        %                         
                        if params.is_sparse_dictionary
                            u = sparsify_dictionary_element(u, params);
                            assert(~nnz(isnan(u)));
                        end
                        %
                        if params.lambda_D ~= 0
                            uj_norm = sqrt((u')*u);
                            if uj_norm == 0
                                coef = 0;
                            else
                                coef = (1-params.lambda_D/uj_norm);
                                assert(~isnan(coef));
                                if coef < 0
                                    coef = 0;
                                end  
                            end
                            %                
                            u = coef*u;
                            %                         
                            if nnz(isnan(u))
                                display 'NaN in D';
                            end
                            clear uj_norm;
                        end
                        %
                        D(:,j) = u*(1/max(1,sqrt(u'*u)));
                    end
                    %
                    max_diff = max(max(abs(Dprev-D)));
                    fprintf('\nmax_diff: %f', max_diff);
                    if max_diff < params.epsilon
                        converged = 1;
                    end 
            end
        case 'Mairal'
            converged = 0;
            curr_count = 0;
            %             
            while ~converged
                curr_count = curr_count + 1;
                if curr_count > max_num_iter
                    break;
                end
                %                 
                Dprev = D;
                for j=1:k
                    if ~A(j,j)
                        a = 1e-100;
                    else
                        a = A(j,j);
                    end
                    %              
                    u =  (B(:,j) - D*A(:,j)) + A(j,j)*D(:,j);
                    %      
                    if ~params.is_sparse_dictionary
                        u = u/a;
                    else
                        if ~all(u == 0)
                            u = u/a;
                            u = sparsify_dictionary_element(u, params);
                        end
                        assert(~nnz(isnan(u)));
                    end
                    %
                    D(:,j) = u*(1/max(1,sqrt(u'*u)));
                end
                %
                max_diff = max(max(abs(Dprev-D)));
                fprintf('\nmax_diff: %f', max_diff);
                if max_diff < params.epsilon
                    converged = 1;
                end 
            end
        case 'GroupMairal'
            converged = 0;
            curr_count = 0;
            %             
            while ~converged
                curr_count = curr_count + 1;
                if curr_count > max_num_iter
                    break;
                end
                %                 
                Dprev = D;
                for j=1:k
                    if ~nnz(D(:,j))  % all-zeros dictionary element - skip it 
                        continue;
                    end
%                     
                    if ~A(j,j)
                        a = 1e-100;
                    else
                        a = A(j,j);  
                    end
%                     
                    % updates for group dictionary learning Bengio 2009
                    z =  A(j,:)*D' - A(j,j)*D(:,j)';
                    if nnz(isnan(z))
                        display 'Nan in z';
                    end 
                    %                 
                    uj =  B(:,j) - z'; 
                    %                 
                    if ~(D(:,j)'*D(:,j))
                        display 'zero norm of D_j' 
                    end
                    %      
                    if ~params.is_sparse_dictionary
                        uj = uj/a;
                    else
                        if ~all(uj == 0)
                            uj = uj/a;
                            uj = sparsify_dictionary_element(uj, params);
                        end
                        assert(~nnz(isnan(uj)));
                    end
                    %
                    if params.is_sparse_dictionary
                        uj_norm = sqrt((uj')*uj);
%                         uj_norm = sum(abs(uj));
                    else
                        uj_norm = sqrt((uj')*uj);
                    end
                    %                 
                    if uj_norm == 0
                        coef = 0;
                    else
                        coef = (1-params.lambda_D/uj_norm);
                        assert(~isnan(coef));
                        if coef < 0
                            coef = 0;
                        end  
                    end
                    %                
                    % soft thresholding: if the last element of dictionary, do
                    % not kill it
                    % if all other elements are zero already, or the dictionary
                    % has just 1 element already
                    d_old = D(:,j);
                    %   
                    D(:,j) = coef*uj;
                    %                 
                    if ~nnz(sum(abs(D)))
                        D(:,j) = d_old;
                    end
                  if nnz(isnan(D(:,j)))
                    display 'NaN in D';
                  end
                  % sahil added code here in regards to the norm of dictionary elements not being too large.
                  curr_dictionary_element_norm = sqrt(D(:,j)'*D(:,j));
                  D(:,j) = D(:,j)*(1/max(1,curr_dictionary_element_norm));
                end
                % 
                max_diff = max(max(abs(Dprev-D)));
%                 
                if ~params.is_sparse_computations
                    fprintf('\nmax_diff: %f', max_diff);
                end
%                 
                if max_diff < params.epsilon
                    converged = 1;
                end
            end
        [~,ind] = find(sum(abs(D)));
        if isempty(ind)
            display 'empty dictionary!'
            pause;
        end
    end
    %
    assert(issparse(D) || (~params.is_sparse_computations));
end

function u = sparsify_dictionary_element(u, params)
    n = length(u); 
    %   
%     tic;
    if strcmp(params.dictionary_element_sparse_algo, 'proximal')
        num_nonzero_dict_element = floor(params.nz_in_dict*n);
        if num_nonzero_dict_element < n
            dict_element_lam = binary_search_proximal_threshold(u, num_nonzero_dict_element, max(0.01*num_nonzero_dict_element, 1));
            u = sign(u).*max(abs(u)-dict_element_lam, 0);
        end
%         fprintf(' dnnz: %d, ', nnz(u));
    elseif strcmp(params.dictionary_element_sparse_algo, 'lars')
        num_nonzero_dict_element = floor(params.nz_in_dict*n);
        u = lars(eye(n), u, 'lars', -num_nonzero_dict_element, 1, eye(n));
        u = u(end,:)'; clear num_nonzero_dict_element;
        assert(size(u, 1) == n); assert(size(u,2) == 1);
    else
        error('invalid algo specification for computing sparse dictionary element.')
    end
    %  
%     compute_time = toc;
%     if compute_time > 0.25
%         fprintf('Time to compute a sparse, LARS based solution, was %f.\n', toc);
%     end
end
