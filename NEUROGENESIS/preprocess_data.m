function X = preprocess_data(X)
% each column in X is a single data point.
% make sure that we are taking mean along correct dimensions
% it should also be consistent across all the data sets processed as the
% original code is still not calling this function.
    X = double(X)/255;
%     
%     X=X-repmat(mean(X),[size(X,1) 1]);
%     X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
% 
    X=X-repmat(mean(X, 2),[1 size(X,2)]);
    X=X ./ repmat(sqrt(sum(X.^2, 2)),[1 size(X,2)]);
