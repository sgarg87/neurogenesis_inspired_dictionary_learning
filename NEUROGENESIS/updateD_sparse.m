function  [D,A,B] = updateD_sparse(D_old,code,x,lambda_D,mu,eta,epsilon,data_type,D_update_method,A,B, data_history,code_history)
    % update dictionary, given the current dictionary, sparse code, data and parameters
    % 
    % lambda_D - regularization parameter for optimizing the dictionary (weight on group l1/l2); 
    %			if 0, no dictionary elements are ever removed
    % eta - learning rate in stochastic gradient descent 
    % data_type - 'Gaussian', or 'Bernoulli', or another exp-family
    % D_update_method 
    % data_history   - all data so far
    % 
    n = size(D_old,1);
    k = size(D_old,2);
    D = D_old;
    %     
    switch D_update_method
    case 'SG' %stochastic gradient with thresholding, i.e. proximal method
        D = stochastic_gradient_update(D, n, k, code,x,lambda_D,mu,epsilon,data_type);
    case 'Mairal'
        D = mairal_update(D, k, epsilon,A,B);
    case 'GroupMairal'
        [D, A, B] = group_mairal_update(D,n,lambda_D,epsilon,A,B);
    end
end

function [D, A, B] = group_mairal_update(D, n, lambda_D,epsilon,A,B)
    % sahil: this is the implementation of equation (11) in the group sparsity paper.
    converged = 0;
    %
    num_nonzero_dict_element = int(n*0.2);
    %     
    while ~converged
        Dprev = D;
        %         
        for j=1:k
            if ~nnz(D(:,j))  % all-zeros dictionary element - skip it 
                continue;
            end

            if ~A(j,j)
                a = 1;
            else
                a = A(j,j);  
            end
            % 
            % updates for group dictionary learning Bengio 2009
            z =  A(j,:)*D' - A(j,j)*D(:,j)';
            assert(~nnz(isnan(z)));
            uj =  B(:,j) - z'; clear z;            
            %             
            uj = larsen(eye(n), uj, a, -num_nonzero_dict_element, 0);
            display(uj);
            uj = uj(end,:)';            
            display(uj);
            assert(false);
            %             
            %% sahil: this doesn't seem right. it should be uj instead of D(:, j) here.
            coef = (1-lambda_D/sqrt((uj)'*uj));
            %             
            if isnan(coef)
                display 'nan coef';
            end
            %                 
            if coef < 0
                coef = 0;
            end  
            % 
            % soft thresholding: if the last element of dictionary, do
            % not kill it
            % if all other elements are zero already, or the dictionary
            % has just 1 element already
            d_old = D(:,j);
            % 
            D(:,j) = (1/a)*coef*uj;
            if ~nnz(sum(abs(D)))
                D(:,j) = d_old;
            end
%               else D(:,j) = u;
%               end
          if nnz(isnan(D(:,j)))
            display 'NaN in D';
          end
          % sahil adding code here in regards to the norm of dictionary
          % elements not being too large. This is assuming that D(:, j) is
          % already updated with the new learning.
          curr_dictionary_element_norm = sqrt(D(:,j)'*D(:,j));
          D(:,j) = D(:,j)*(1/max(1,curr_dictionary_element_norm));
        end
        % sahil: commenting the line below as it should be printed only if debugging the code.            
%             max(max(abs(Dprev-D)))            
        if max(max(abs(Dprev-D))) < epsilon
            converged = 1;
        end
    end    
    % 
    [nzD,ind] = find(sum(abs(D)));
    if ~length(ind) 
        display 'empty dictionary!'
        pause;
    end
end

function D = mairal_update(D, k, epsilon,A,B)
    converged = 0;
    while ~converged
        Dprev = D;
        for j=1:k
            if ~A(j,j)
                a = 1;
            else a = A(j,j);
            end
            u =  (B(:,j) - D*A(:,j))/a + D(:,j);
            D(:,j) = u*(1/max(1,sqrt(u'*u)));
        end
        %             
        if max(max(abs(Dprev-D))) < epsilon
            converged = 1;
        end 
    end
end

function D = stochastic_gradient_update(D, n, k, code,x,lambda_D,mu,epsilon,data_type)
    %%%%%%%%%%%%%%%%%%%%%% this modification  is NOT an SG approach - looks at ALL data
          %  x = data_history; 
          %  code = code_history; 
           % instead of stochastic gradient (optimizing w.r.t. recent data batch), 
                              % try all data samples seen so far - to optimize
                              % same objective as Mairal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 
    converged = 0;
    while ~converged
    %for iter = 1:4
        Dprev = D;
    %  
    % compute the step size eta for gradient step - use 1/L, where L is an upper bound on
    % the largest eigenvalue
    %  
    [U,S,V] = svd(code*code');
    L = S(1,1)+0.01;
    eta = 1/L;
    % gradient step; first index - input dimension, second - dictionary element
            % sahil: "link_func(D*code,data_type) - x" is negative of error we make in inferring x.     
            D = D - eta*(link_func(D*code,data_type) - x)*code';
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
                coef = 1-lambda_D/sqrt(D(:,j)'*D(:,j));
                if coef < 0
                    coef = 0;
                end
                d_old = D(:,j);

                D(:,j) = D(:,j)*coef;

                if ~nnz(sum(abs(D)))
                    D(:,j) = d_old;
                end
            end
            % 
            if max(max(abs(Dprev-D))) < epsilon
                converged = 1;
            end   
    end
end






 