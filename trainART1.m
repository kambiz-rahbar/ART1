% kambiz.rahbar@gmail.com, 18 Jan 2020

function [W, T] = trainART1(trainData, C, r)
% C: number of clusters
% r:‫‪ vigilance‬‬ ‫‪parameter‬‬
% W: returned weights (bottom-up weight)
% T: returned indices (top-down weight)



% M: number of elements in one data vector
% N: number of data vectors
[M, N] = size(trainData);

W = 0.2*ones(M, C); % pre-init weights (bottom-up weight)
T = ones(M, C);     % pre-init indices (top-down weight)

for n = 1:N
    X = trainData(:,n);
    X_new = W'*X;
    [~, j] = max(X_new);
    while(1)
        S = X'*T(:,j) / sum(X);
        if (S > r || j == C)   % data belongs to the existing cluster
            break;
        else
            j = j+1; % make a new cluster
        end
    end
    W(:,j) = (T(:,j) .* X) / (0.5 + X' * T(:,j));  % update weights
    T(:,j) = T(:,j) .* X;                          % update indices
end
