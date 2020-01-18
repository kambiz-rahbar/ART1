% kambiz.rahbar@gmail.com, 18 Jan 2020

function [W, T] = trainART1(trainData, C, r, fast_learning)
% C: number of clusters
% r:‫‪ vigilance‬‬ ‫‪parameter‬‬
% fast_learning: {1: fast learning / 0: clasic learning}
% W: returned weights (bottom-up weight)
% T: returned indices (top-down weight)

if ~fast_learning
    B = 0.1;  % learning rate
end

% M: number of elements in one data vector
% N: number of data vectors
[M, N] = size(trainData);

W = 1/(M+1)*ones(M, C); % pre-init weights (bottom-up weight)
T = ones(M, C);     % pre-init indices (top-down weight)

for n = 1:N
    X = trainData(:,n);
    Output = W'*X;
    [~, j] = max(Output);
    while(1)
        S = X'*T(:,j) / sum(X); % measure similarity
        if (S > r || j == C)    % data belongs to the existing cluster
            break;
        else
            j = j+1; % make a new cluster
        end
    end
    W(:,j) = (T(:,j) .* X) / (0.5 + T(:,j)' * X);  % update weights
    if fast_learning
        T(:,j) = T(:,j) .* X;                   % fast update indices
    else
        B = B * 0.99;                           % update learning rate
        T(:,j) = (1-B)*T(:,j) + B* T(:,j).* X;  % classic update indices
    end
end
