function  [D,A,B] = updateD(D_old,code,x,params,D_update_method,A,B)
    % update dictionary, given the current dictionary, sparse code, data and parameters
    % D_update_method 
    % 
    n = size(D_old,1);
    % 
    if params.is_sparse_dictionary
        num_nonzero_dict_element = params.nz_in_dict*n;
        if num_nonzero_dict_element < 1
            num_nonzero_dict_element = 1;
        else
            num_nonzero_dict_element = round(num_nonzero_dict_element);
            if num_nonzero_dict_element > n
                num_nonzero_dict_element = n;
            end
        end
    else
        num_nonzero_dict_element = [];
    end
    % 
    k = size(D_old,2);
    D = D_old;
    % 
    switch D_update_method
        case 'SG' %stochastic gradient with thresholding, i.e. proximal method
    %         assert(~is_sparse_dictionary);

    %%%%%%%%%%%%%%%%%%%%%% this modification  is NOT an SG approach - looks at ALL data
          %  x = data_history; 
          %  code = code_history; 
           % instead of stochastic gradient (optimizing w.r.t. recent data batch), 
                              % try all data samples seen so far - to optimize
                              % same objective as Mairal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            converged = 0;
            while ~converged
            %for iter = 1:4
                Dprev = D;

    %max(max(D));
            % compute the step size eta for gradient step - use 1/L, where L is an upper bound on
            % the largest eigenvalue
    %  
            [U,S,V] = svd(code*code');
            L = S(1,1)+0.01;
            eta = 1/L;
            % gradient step; first index - input dimension, second - dictionary element
                    % sahil: "link_func(D*code,data_type) - x" is negative of error we make in inferring x.     
                    D = D - eta*(link_func(D*code,params.data_type) - x)*code';

                    if nnz(isinf(D))
                        display 'infty in D'
                        pause;
                    end
                    if nnz(isnan(D))
                        display 'NaNs in D'
                        pause;
                    end

                    for j=1:k
                        for jj = 1:n  % first, soft thresholding of individual edges 
                            if ~D(jj,j) % zero element  - skip it
                                continue;
                            end
                            coef = 1 - mu/abs(D(jj,j));%  mu*abs(D(jj,j)); - does not look right, must be / not *
                            if coef < 0 
                                coef = 0;
                            end
                            D(jj,j) = D(jj,j)*coef;
                        end
                        if ~nnz(D(:,j))  % all-zeros dictionary element - skip it 
                            continue;
                        end
    %                     
    %                     if ~nnz(D(:,j))
    %                         display 'all zeros dictionary element'
    %                         pause;
    %                     end
                        coef = 1-params.lambda_D/sqrt(D(:,j)'*D(:,j));
                        %                     
                        if coef < 0
                            coef = 0;
                        end
                        d_old = D(:,j);

                        D(:,j) = D(:,j)*coef;

                        if ~nnz(sum(abs(D)))
                            D(:,j) = d_old;
                        end
               %         norm(D(:,j))


                        %A(:,j) = A(:,j)*coef; %not sure we need this here - SG does not use A and B!
                        %B(:,j) = B(:,j)*coef;
                    end

                    if max(max(abs(Dprev-D))) < epsilon
                        converged = 1;
                    end   

                    %  let's normalize the dictionary - EM diverges, not a good idea
                    %%D = normalize(D);
            end
        case 'Mairal'
            converged = 0;
            while ~converged
                Dprev = D;
                for j=1:k
                    if ~A(j,j)
                        a = 1e-30;
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
                            u = lars(eye(n), u, 'lars', -num_nonzero_dict_element, 1, eye(n));
                            u = u(end,:)';
                            assert(size(u, 1) == n); assert(size(u,2) == 1);
                            u = u/a;                        
                        end
                        assert(~nnz(isnan(u)));
                    end
                    %
                    D(:,j) = u*(1/max(1,sqrt(u'*u)));
                end
                %             
                if max(max(abs(Dprev-D))) < params.epsilon
                    converged = 1;
                end 
            end
        case 'GroupMairal'
            converged = 0;  
            while ~converged
                Dprev = D;
                for j=1:k
                    if ~nnz(D(:,j))  % all-zeros dictionary element - skip it 
                        continue;
                    end
                    if ~A(j,j)
                        a = 1e-30;
                    else
                        a = A(j,j);  
                    end
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
                            uj = lars(eye(n), uj, 'lars', -num_nonzero_dict_element, 1, eye(n));
                            uj = uj(end,:)';
                            assert(size(uj, 1) == n); assert(size(uj,2) == 1);
                            uj = uj/a;
                        end
                        assert(~nnz(isnan(uj)));
                    end
                    %
                    if params.is_sparse_dictionary
                        uj_norm = sum(abs(uj));
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
                if max(max(abs(Dprev-D))) < params.epsilon
                    converged = 1;
                end
            end
        [~,ind] = find(sum(abs(D)));
        if isempty(ind)
            display 'empty dictionary!'
            pause;
        end
    end
end
