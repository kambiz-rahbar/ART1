% kambiz.rahbar@gmail.com, 18 Jan 2020

function [resultCluster] = testART1(testData, W, T, r)
% W: weights (bottom-up weight)
% T: indices (top-down weight)
% r:‫‪ vigilance‬‬ ‫‪parameter‬‬
% C: returned cluster


[~, N] = size(testData); % N: number of data vectors
[~, C] = size(W);        % N: number of clusters

resultCluster = zeros(1, N);
for n = 1:N
    X = testData(:,n);
    X_new = W'*X;
    [~, j] = max(X_new);
    while(1)
        S = X'*T(:,j) / sum(X);
        if (S > r || j == C)  % check if the data belongs to the current cluster
            break;
        else
            j = j+1; % check the next cluster
        end
    end
    resultCluster(n) = j;
end