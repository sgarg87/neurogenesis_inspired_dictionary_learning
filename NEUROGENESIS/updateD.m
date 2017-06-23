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
%             if params.is_sparse_dictionary
%                 warning('Sparse learning of dictionary elements is not implemented for SG method yet');
%             end
    %         assert(~is_sparse_dictionary);

    %%%%%%%%%%%%%%%%%%%%%% this modification  is NOT an SG approach - looks at ALL data
          %  x = data_history; 
          %  code = code_history; 
           % instead of stochastic gradient (optimizing w.r.t. recent data batch), 
                              % try all data samples seen so far - to optimize
                              % same objective as Mairal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %               
%             max_num_iter = 100;
            converged = 0;
            curr_count = 0;
            %             
            while ~converged
                curr_count = curr_count + 1;
                %                 
                if curr_count > max_num_iter
                    break;
                end

            %for iter = 1:4
                Dprev = D;

    %max(max(D));
%     
% 
% 
% Sahil commented the code stochastic rate selection with SVD, too
% expensive, using an alternative like a constant rate instead.
%             % compute the step size eta for gradient step - use 1/L, where L is an upper bound on
%             % the largest eigenvalue
%             tic;
%             [U,S,V] = svd(code*code');
%             L = S(1,1)+0.01;
%             eta = 1/L;            
%             toc
%             eta
            %  
            %     
                eta = params.eta;
            % 
            %             
            % 
%             % gradient step; first index - input dimension, second - dictionary element
%                 % sahil: "link_func(D*code,data_type) - x" is negative of error we make in inferring x.
%                 D = D - eta*(link_func(D*code,params.data_type) - x)*code';
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
%                 if params.is_sparse_dictionary || (params.lambda_D ~= 0)
                    for j=1:k
                        %                         
                        % gradient step; first index - input dimension, second - dictionary element
                        % sahil: "link_func(D*code,data_type) - x" is negative of error we make in inferring x.
                        % todo: make this step more efficient by computing the change only for the current block (jth dictionary element)
                        D_change = (link_func(D*code,params.data_type) - x)*code';
                        u = D(:, j) - eta*D_change(:, j);
                        % 
%                         u = D(:, j);
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
%                 end
% 
% 
%  Sahil commented the code block, dealing with sparsity of elements and
%  killing
%                     for j=1:k
%                         for jj = 1:n  % first, soft thresholding of individual edges 
%                             if ~D(jj,j) % zero element  - skip it
%                                 continue;
%                             end
%                             coef = 1 - params.mu/abs(D(jj,j));%  mu*abs(D(jj,j)); - does not look right, must be / not *
%                             if coef < 0 
%                                 coef = 0;
%                             end
%                             D(jj,j) = D(jj,j)*coef;
%                         end
%                         if ~nnz(D(:,j))  % all-zeros dictionary element - skip it 
%                             continue;
%                         end
%     %                     
%     %                     if ~nnz(D(:,j))
%     %                         display 'all zeros dictionary element'
%     %                         pause;
%     %                     end
%                         coef = 1-params.lambda_D/sqrt(D(:,j)'*D(:,j));
%                         %                     
%                         if coef < 0
%                             coef = 0;
%                         end
%                         d_old = D(:,j);
% 
%                         D(:,j) = D(:,j)*coef;
% 
%                         if ~nnz(sum(abs(D)))
%                             D(:,j) = d_old;
%                         end
%                %         norm(D(:,j))
% 
% 
%                         %A(:,j) = A(:,j)*coef; %not sure we need this here - SG does not use A and B!
%                         %B(:,j) = B(:,j)*coef;
%                     end

                    max_diff = max(max(abs(Dprev-D)));
                    fprintf('\nmax_diff: %f', max_diff);
                    if max_diff < params.epsilon
                        converged = 1;
                    end 
%                     if max(max(abs(Dprev-D))) < params.epsilon
%                         converged = 1;
%                     end   
                    % 
                    %  let's normalize the dictionary - EM diverges, not a good idea
                    %%D = normalize(D);
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
