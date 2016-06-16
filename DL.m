function [D,A, B, err,correl_all] = DL(data,D0,nonzero_frac,lambda_D,mu,eta,epsilon,T,new_elements,data_type,D_update_method, A, B)
    % 
    % learn a dictionary D, and a  sparse code C, for data
    % 
    % parameters:
    % nonzeros_C - regularization parameter for optimizing sparse code C: for LARS, # of nonzeros
    %              if nonzeros = -1, the sparsity level is selected automatically as  best-fitting for each given sample
    % lambda_D - regularization parameter for optimizing the dictionary (weight on group l1/l2); 
    %			if 0, no dictionary elements are ever removed
    % eta - learning rate in stochastic gradient descent 
    % T  - number of iterations
    % new_elements - the  number of new dictionary elements to generate per each sample (if 0, no neurogen occuring)
    % data_type - 'Gaussian', or 'Bernoulli', or another exp-family
    %
    %
    %     
    n = size(D0,1);
    k = size(D0,2);
    %
    D = D0;
    C = [];
    %
    % reset the ?past? information
    assert((isempty(A) && isempty(B)) || ((~isempty(A)) && (~isempty(B))));
    if isempty(A) && isempty(B)
        A = zeros(k,k); B = zeros(n,k); % matrices used by Mairal's dictionary update method
    else
        assert(size(A, 1) == k); assert(size(A, 2) == k);
        assert(size(B, 1) == n); assert(size(B, 2) == k);
    end
    %
    batch_size = 20;
    %
    [n1, n2]=size(data);
    t_start=1;
    t_end=batch_size;
    t = 0; % iteration

    if (find(mean(data) == 0))
        display 'all zero column in data';
        pause;
    end

    while t_end <= T  % up to T samples
        t = t+1;
        %  get the next batch of input samples 
        x = data(:,t_start:t_end);
        data_history = data(:,1:t_end);
        % 
        t_start=t_end+1;
        t_end = t_end + batch_size;
        %                      
        % 1. neurogenesis step
        % sahil: this is to account for the case of random dictionary. 
        % in this case, we don't really learn. ideally, we should avoid
        % such cases in this function.         
        if  new_elements < 0   % just use the initial dictionary
            D = D0;
        end
        % changed on 11/9
        % evaluate the current dictionary before adding random elements
        %
        tic;
        %        
        %         
        [code,err(t,:),correl] = sparse_coding(x,D,nonzero_frac,data_type);
        correl_S(t,:) = correl(1,:);
        correl_P(t,:) = correl(2,:);
        pre_err = err; pre_correl = correl; 
        pre_correl_P(t,:)=correl(1,:); pre_correl_S(t,:)=correl(2,:);
        fprintf('Number of seconds to do sparse coding was %f.\n', toc);    
        %          
        % sahil: not doing any learning.        
        if new_elements <0 % initial dictionary    
            continue;
        end
        %  
        % neurogen version 
        
        %% sahil commented this code block as it is exactly same computation happening above.
        % IR - changes for conditional neurogenesis 6/15
        % % 1. first, sparse-code with old dictionary, keep the performance scores
        % [code,err(t,:),correl] = sparse_coding(x,D,nonzero_frac,data_type);
        % correl_S(t,:) = correl(1,:);
        % correl_P(t,:) = correl(2,:);     
        % pre_err = err; pre_correl = correl; 
        % pre_correl_P(t,:)=correl(1,:); pre_correl_S(t,:)=correl(2,:);
        % 
        if t > 1  & new_elements > 0  % if this is not the first iteration and neurogen should be happening (new_elements >0)
          % perform error check and increase or decrease neurogen     
          % if this new, 'test set' error on a new batch of samples is 'much'
            % worse than the 'train'/post error on the previous batch (i.e.
            % generalization is bad), then increase neurogenesis rate
      
          %rel_err = (norm(pre_err(t,:))-norm(post_err(t-1,:)))/norm(post_err(t-1,:));
          rel_corr = mean(post_correl_P(t-1,:))-mean(pre_correl_P(t,:)); %/mean(post_correl_P(t,:)));
          % 
          % sahil commented: epsilon is passed as a parameter above.
          % epsilon = 0.1;
          % 
          birth_rate=0;
          if  rel_corr > epsilon   %'generalization factor' : current test error vs. previous train error
                % increase neurogen (unless depression factor is > 0 :)
                birth_rate = 1;%1.5*(1+rel_corr);
          end
          % 
          new_elements1 = floor(birth_rate*new_elements); % may play with increasing rate
              if new_elements1  % conditional addition of new elements
                fprintf('Adding %d new elements.', new_elements1);
                D = [D normalize(rand(n,new_elements1))]; 
                B = [B zeros(n,new_elements1)];
                A = [A zeros(k,new_elements1)];  
                A = [A;zeros(new_elements1,k+new_elements1)];
                 % sahil: added code lines below to update C (C is not used currently except for debugging).
                if ~isempty(C)
                % display(new_elements);
                    C(k+1:k+new_elements, :) = 0;
                end
                % sahil code ends here.
                
                k = k + new_elements1;
              end
        end

        % 
        % 2. sparse coding step AFTER adding random elements,to use it in dict update for each element in the data batch
        tic;
        [code] = sparse_coding(x,D,nonzero_frac,data_type);
        fprintf('Number of seconds to do sparse coding was %f.\n', toc);
        % 
        % matrices used by Mairal's dictionary update method     
        A = A + code*code';  
        B = B + x*code';
        % sahil: adding these code to the matrix to keep track of all the learning together.
        C = [C code];
        % sahil code ends here.     
        % 	
        if nnz(isnan(A))
            display 'A is NaN';
            pause;
        end
        %         
        % 3. dictionary learning step
        %    currently, implemented as 'truncated' stochastic gradient, or proximal method with group sparsity
        %     
        % just in case, giving all previous data and current encoding to updateD
        % sahil: checked that we don't really use the code_history. so, for now, should we comment the code below.    
        % sahil commented the line below and instead initilizing code_history to empty.
        % [code_history] = sparse_coding(data_history,D,nonzero_frac,data_type);
        code_history = [];
        % 
        tic;
        [D,A,B] = updateD(D,code,x,lambda_D,mu,eta,epsilon,data_type,D_update_method, A,B, data_history,code_history) ;
        fprintf('Number of seconds to update the dictionary was %f.\n', toc);
        %    
%         display(sum(abs(D)));
        [nzD,ind] = find(sum(abs(D)));
        % active neurons.    
        num_non_sparse_dictionary_elements = length(ind);
        % Number of non-sparse dictionary elements.
        fprintf('Number of non-sparse dictionary elements are %d.\n', num_non_sparse_dictionary_elements);
        clear num_non_sparse_dictionary_elements;
        zero_idx = setdiff(1:size(D, 2), ind);
        %     
        if length(zero_idx)
            display('The indices of the killed dictionary elements are:');
            display(zero_idx);
            clear zero_idx;
        else
            display('No elements killed.')
        end
        %     
        if ~length(ind)
            display 'empty dictionary!'
            pause;
        else
                D = D(:,ind);
                B = B(:,ind);
                A = A(:,ind); A = A(ind,:);
                k = length(ind);
                %
                % sahil: adding code line below for updating C (C is used only for debugging as of now).
                C = C(ind, :);
                % sahil code ends here.             
        end
        %   change added for conditional neurogen -IR
        [code,post_err(t,:),post_correl] = sparse_coding(x,D,nonzero_frac,data_type);
        post_correl_S(t,:) = post_correl(1,:);
        post_correl_P(t,:) = post_correl(2,:);
    end

    %   change added for conditional neurogen -IR  
    [er,ec] = size(err);
    err = reshape(err',1,er*ec);

    correl_all(1,:) = reshape(correl_P',1,er*ec);
    correl_all(2,:) = reshape(correl_S',1,er*ec);    
    % 
    %err = mean(err')';
    %correl = mean(correl')';
end


