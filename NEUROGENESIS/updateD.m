function  [D,A,B] = updateD(D_old,code,x,lambda_D,mu,eta,epsilon,data_type,D_update_method,A,B, data_history,code_history)

% update dictionary, given the current dictionary, sparse code, data and parameters

% lambda_D - regularization parameter for optimizing the dictionary (weight on group l1/l2); 
%			if 0, no dictionary elements are ever removed
% eta - learning rate in stochastic gradient descent 
% data_type - 'Gaussian', or 'Bernoulli', or another exp-family
% D_update_method 
% data_history   - all data so far

n = size(D_old,1);
k = size(D_old,2);
D = D_old;
    
% 
switch D_update_method
    case 'SG' %stochastic gradient with thresholding, i.e. proximal method

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
 
        [U,S,V] = svd(code*code');
        L = S(1,1)+0.01;
        eta = 1/L;
        % gradient step; first index - input dimension, second - dictionary element
                % sahil: "link_func(D*code,data_type) - x" is negative of error we make in inferring x.     
                D = D - eta*(link_func(D*code,data_type) - x)*code';
                
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
                    
%                     if ~nnz(D(:,j))
%                         display 'all zeros dictionary element'
%                         pause;
%                     end
                    
                    coef = 1-lambda_D/sqrt(D(:,j)'*D(:,j));
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
        % sahil: this is the standard implementation of the pseudo code in Algo 2 in the paper Mairal et al.
        % challenge is understanding why this algo should work.        
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
            if max(max(abs(Dprev-D))) < epsilon
                converged = 1;
            end 
        end
    case 'GroupMairal'
        % sahil: this is the implementation of equation (11) in the group sparsity paper.
        % sahil: important is how we come up with formulation exactly.
        converged = 0;  
        while ~converged
            Dprev = D;
            for j=1:k
                if ~nnz(D(:,j))  % all-zeros dictionary element - skip it 
                    continue;
                end
            
                if ~A(j,j)
                    a = 1;
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
                coef = (1-lambda_D/sqrt(D(:,j)'*D(:,j)));
                if isnan(coef)
                    display 'nan coef';
                end;
%                 
                if coef < 0
                    coef = 0;
                end  
                
                % soft thresholding: if the last element of dictionary, do
                % not kill it
                % if all other elements are zero already, or the dictionary
                % has just 1 element already
                d_old = D(:,j);
            
                D(:,j) = (1/a)*coef*uj;
                if ~nnz(sum(abs(D)))
                    D(:,j) = d_old;
                end
  %               else D(:,j) = u;
  %               end
              if nnz(isnan(D(:,j)))
                display 'NaN in D';
              end

            end

            % sahil: commenting the line below as it should be printed only if debugging the code.            
%             max(max(abs(Dprev-D)))
            
            if max(max(abs(Dprev-D))) < epsilon
                converged = 1;
            end


        end
        
%         for j=1:k
% %           for jj = 1:n  % first, soft thresholding of individual edges 
% %             coef = 1 - mu*abs(D(jj,j));
% %             if coef < 0 
% %                 coef = 0;
% %             end
% %             D(jj,j) = D(jj,j)*coef;
% %           end
%                    
%           coef = 1-lambda_D/sqrt(D(:,j)'*D(:,j));
%           
%           if isnan(coef)
%                     display 'coef is NaN';
%                     pause;
%           end
%                 
%           if coef < 0
%             coef = 0;
%           end
%           D(:,j) = D(:,j)*coef;
%           A(:,j) = A(:,j)*coef;   
%           B(:,j) = B(:,j)*coef;
%       
%         end
  
           
    [nzD,ind] = find(sum(abs(D)));
    if ~length(ind) 
        display 'empty dictionary!'
        pause;
    end
        


end





 