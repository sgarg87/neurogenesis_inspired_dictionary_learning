function [D,A, B, err,correl_all] = DL(data, D0, params, D_update_method, A, B)
    % Note: if the input data is sparse, be very careful when normalizing
    % dictionary elements, or normalizing/centering data.
    %     
    % learn a dictionary D, and a  sparse code C, for data
    %
    display('xxxxxxxxxxxxxxxxxxxxx DL start xxxxxxxxxxxxxxxxxxxxxxxx');
    fprintf('Initial number of dictionary elements is %d.\n', size(D0, 2));
    display('.......................................................');
    %
    %
%     if ~issparse(data) && params.is_sparse_computations
%         data = sparse(data);
%     end
    %
    %     
    n = size(D0,1);
    k = size(D0,2);
    %
    D = D0;
    C = [];
    %
    % reset the ?past? information
    %  
    %     
    if isempty(A)
        A = zeros(k,k); % matrices used by Mairal's dictionary update method
    else
        assert(size(A, 1) == k); assert(size(A, 2) == k);
    end
    % 
    if isempty(B)
        B = zeros(n,k); % matrices used by Mairal's dictionary update method
    else
        assert(size(B, 1) == n); assert(size(B, 2) == k);
    end
    %     
    %     
    t_start=1;
    t_end=params.batch_size;
    t = 0; % iteration

%     Sahil commented the code.
%     if (find(mean(data) == 0))
%         display 'all zero column in data';
%         pause;
%     end
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
        [code,err(t,:),correl] = sparse_coding(x,D,params);
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
%                     Sahil added a simpler condition for the conditional
%                     neurogenesis (if low correlation, add neurons).
                    curr_error_fr_conditional_neurogenesis = 1-mean(correl(1,:));
%                     display(curr_error_fr_conditional_neurogenesis);
                    if curr_error_fr_conditional_neurogenesis >  params.errthresh
                        birth_rate = curr_error_fr_conditional_neurogenesis;
                    end
                    clear curr_error_fr_conditional_neurogenesis;
% 
% 
% 
%                     % perform error check and increase or decrease neurogen
%                     % if this new, 'test set' error on a new batch of samples is 'much'
%                     % worse than the 'train'/post error on the previous batch (i.e.
%                     % generalization is bad), then increase neurogenesis rate
%                     % rel_err = (norm(pre_err(t,:))-norm(post_err(t-1,:)))/norm(post_err(t-1,:));
%                     rel_corr = mean(post_correl_P(t-1,:))-mean(pre_correl_P(t,:)); %/mean(post_correl_P(t,:)));
%                     if  rel_corr > params.errthresh   %'generalization factor' : current test error vs. previous train error
%                         % increase neurogen (unless depression factor is > 0 :)
%                         birth_rate = 1;%1.5*(1+rel_corr);
%                     end
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
                %% not sure if the normalization being done on right dimensions.
                %% is normalization really required here ?       
                %                 
                % sahil: added code for grand mother neurons
                new_Dict_elements = normalize(rand(n,curr_new_elements));
                if params.is_grand_mother_neurons
                    mairal_params = params;
                    mairal_params.lambda_D = 0;
                    mairal_params.new_elements = 0;
                    mairal_params.T = size(x, 2);
                    [new_Dict_elements,~, ~, ~,~] = DL(x, new_Dict_elements, mairal_params, 'Mairal', [], []);
                    clear mairal_params;
                end
                %      
                fprintf('Adding %d new elements.', curr_new_elements);
                D = [D new_Dict_elements];
                %
                if params.is_immunized_born_neurons                    
                    B = [B params.immunization_dose_fr_born_neurons*ones(n,curr_new_elements)];
                    A = [A params.immunization_dose_fr_born_neurons*ones(k,curr_new_elements)];
                    A = [A; params.immunization_dose_fr_born_neurons*ones(curr_new_elements,k+curr_new_elements)];                    
                else
                    B = [B zeros(n,curr_new_elements)];
                    A = [A zeros(k,curr_new_elements)];  
                    A = [A;zeros(curr_new_elements,k+curr_new_elements)];
                end
                %
                if ~isempty(C)
                    C(k+1:k+curr_new_elements, :) = 0;
                end
                k = k + curr_new_elements;
                %                 
                % sparse coding step AFTER adding random elements,to use it in dict update for each element in the data batch
                tic;
                [code] = sparse_coding(x,D,params);
                fprintf('Number of seconds to do sparse coding was %f.\n', toc);
            end
        end
        %
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
        tic;
        D = updateD(D, code, x, params, D_update_method, A, B);
        %         
        fprintf('\nNumber of seconds to update the dictionary was %f.\n', toc);
        %
        if (strcmp(D_update_method, 'GroupMairal')) || (strcmp(D_update_method, 'SG') && (params.lambda_D ~= 0))
            [~,ind] = find(sum(abs(D)));
            % active neurons.
            num_non_sparse_dictionary_elements = length(ind);
            % Number of non-sparse dictionary elements.
            fprintf('Number of non-zero dictionary elements are %d.\n', num_non_sparse_dictionary_elements);
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
        else
            if params.is_reinitialize_dictionary_fixed_size
                % sahil added code block for random initialization of zero columns (non-group sparsity) 
                % and then relearn (as suggested in Mairal 2009).
                [~,zero_idx] = find(~sum(abs(D)));
                if ~isempty(zero_idx)
                    display('Random initialization of zero columns and then relearning.');
                    D(:, zero_idx) = normalize(rand(n,length(zero_idx)));
                    A = A - code*code';
                    B = B - x*code';
                    [code] = sparse_coding(x,D,params);
                    A = A + code*code';
                    B = B + x*code';
                    D = updateD(D, code, x, params, D_update_method, A, B);
                end
            end
        end
        %
        if params.is_conditional_neurogenesis
            [~,~,post_correl] = sparse_coding(x,D,params);
            post_correl_P(t,:) = post_correl(2,:);
        end
    end
    %
    %
    [~,ind] = find(sum(abs(D)));
    display('.......................................................');
    fprintf('Number of non-zero dictionary elements are %d.\n', length(ind));
    display('xxxxxxxxxxxxxxxxxxxxxxx DL end xxxxxxxxxxxxxxxxxxxxxxxxxxx');
    %
    %
    [er,ec] = size(err);
    err = reshape(err',1,er*ec);
    % 
    correl_all(1,:) = reshape(correl_P',1,er*ec);
    correl_all(2,:) = reshape(correl_S',1,er*ec);
end
