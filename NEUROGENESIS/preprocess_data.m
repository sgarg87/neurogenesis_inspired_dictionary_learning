function X = preprocess_data(X)
% each column in X is a single data point.
% make sure that we are taking mean along correct dimensions
%   
% reverting to the original mean and std across dimensions instead of
% across data.
% mean and std across the dimenions (original)
    X=X-repmat(mean(X),[size(X,1) 1]);
    X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
% 
% 
% 
% 
% 
%     % mean and std across data
%     X=X-repmat(mean(X, 2),[1 size(X,2)]);
%     X=X ./ repmat(sqrt(sum(X.^2, 2)),[1 size(X,2)]);
