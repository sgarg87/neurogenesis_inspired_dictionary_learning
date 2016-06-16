function [D,A, B, err,correl_all] = DL(data, D0, params, D_update_method, A, B)
    % learn a dictionary D, and a  sparse code C, for data
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
    t_start=1;
    t_end=params.batch_size;
    t = 0; % iteration

    if (find(mean(data) == 0))
        display 'all zero column in data';
        pause;
    end
    % 
    while t_end <= params.T  % up to T samples
        t = t+1;
        %  get the next batch of input samples 
        x = data(:,t_start:t_end);
%         data_history = data(:,1:t_end);
        % 
        t_start=t_end+1;
        t_end = t_end + params.batch_size;
        %     
        % 1. neurogenesis step
        if  params.new_elements < 0   % just use the initial dictionary
            D = D0;
        end
        % evaluate the current dictionary before adding random elements
        tic;
        [code,err(t,:),correl] = sparse_coding(x,D,params.nonzero_frac);
        correl_S(t,:) = correl(1,:);
        correl_P(t,:) = correl(2,:);
        pre_err = err; pre_correl = correl; 
        pre_correl_P(t,:)=correl(1,:); pre_correl_S(t,:)=correl(2,:);
        fprintf('Number of seconds to do sparse coding was %f.\n', toc);
        %          
        if params.new_elements <0 % initial dictionary    
            continue;
        end
        %  
        if params.new_elements > 0
            if params.is_conditional_neurogenesis
                % if this is not the first iteration and neurogen should be happening (new_elements > 0)
                if t > 1
                    % perform error check and increase or decrease neurogen
                    % if this new, 'test set' error on a new batch of samples is 'much'
                    % worse than the 'train'/post error on the previous batch (i.e.
                    % generalization is bad), then increase neurogenesis rate
                    % rel_err = (norm(pre_err(t,:))-norm(post_err(t-1,:)))/norm(post_err(t-1,:));
                    rel_corr = mean(post_correl_P(t-1,:))-mean(pre_correl_P(t,:)); %/mean(post_correl_P(t,:)));
                    if  rel_corr > params.epsilon   %'generalization factor' : current test error vs. previous train error
                        % increase neurogen (unless depression factor is > 0 :)
                        birth_rate = 1;%1.5*(1+rel_corr);
                    end
                else
                    birth_rate = 0;
                end
                %
                curr_new_elements = floor(birth_rate*params.new_elements);
            else
                curr_new_elements = params.new_elements;
            end
            % neurogen version
            if curr_new_elements > 0
                fprintf('Adding %d new elements.', curr_new_elements);
                D = [D normalize(rand(n,curr_new_elements))];
                B = [B zeros(n,curr_new_elements)];
                A = [A zeros(k,curr_new_elements)];  
                A = [A;zeros(curr_new_elements,k+curr_new_elements)];
                %                 
                if ~isempty(C)
                    C(k+1:k+curr_new_elements, :) = 0;
                end
                k = k + curr_new_elements;
                % sparse coding step AFTER adding random elements,to use it in dict update for each element in the data batch
                tic;
                [code, ~, ~] = sparse_coding(x,D,params.nonzero_frac);
                fprintf('Number of seconds to do sparse coding was %f.\n', toc);
            end
        end
        % 
        % matrices used by Mairal's dictionary update method     
        A = A + code*code';  
        B = B + x*code';
        %         
        C = [C code];
        % 	
        if nnz(isnan(A))
            display 'A is NaN';
            pause;
        end
        %         
        % 3. dictionary learning step
        %    currently, implemented as 'truncated' stochastic gradient, or proximal method with group sparsity
        % just in case, giving all previous data and current encoding to updateD
        % [code_history] = sparse_coding(data_history,D,nonzero_frac,data_type);
%         code_history = [];
        % 
        % 
        tic;
        [D, A, B] = updateD(D, code, x, params, D_update_method, A, B);
        fprintf('Number of seconds to update the dictionary was %f.\n', toc);
        [~,ind] = find(sum(abs(D)));
        % active neurons.
        num_non_sparse_dictionary_elements = length(ind);
        % Number of non-sparse dictionary elements.
        fprintf('Number of non-sparse dictionary elements are %d.\n', num_non_sparse_dictionary_elements);
        clear num_non_sparse_dictionary_elements;
        zero_idx = setdiff(1:size(D, 2), ind);
        % 
        if ~isempty(zero_idx)
            display('The indices of the killed dictionary elements are:');
            display(zero_idx);
            clear zero_idx;
        else
            display('No elements killed.')
        end
        %     
        if isempty(ind)
            display 'empty dictionary!'
            pause;
        else
                D = D(:,ind);
                B = B(:,ind);
                A = A(:,ind); A = A(ind,:);
                k = length(ind);
                %
                C = C(ind, :);
        end
        % 
        if params.is_conditional_neurogenesis
            [~,~,post_correl] = sparse_coding(x,D,params.nonzero_frac);
            post_correl_P(t,:) = post_correl(2,:);
        end
    end
    % 
    [er,ec] = size(err);
    err = reshape(err',1,er*ec);
    % 
    correl_all(1,:) = reshape(correl_P',1,er*ec);
    correl_all(2,:) = reshape(correl_S',1,er*ec);
end