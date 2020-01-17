% kambiz.rahbar@gmail.com, 18 Jan 2020

clc
clear

trainData = [1 1 0 1 1;
             0 1 1 1 0;
             1 0 1 1 1;
             0 0 0 0 0];
    
% trainData = [1 0 1 0 1;
%              1 0 0 0 1;
%              0 1 1 0 0;
%              0 1 1 1 1;
%              0 1 1 1 1;
%              0 1 1 1 1;
%              1 0 0 0 0];

C = 3;   % number of clusters
r = 0.7; %‫‪ vigilance‬‬ ‫‪parameter‬‬
[W, T] = trainART1(trainData, C, r);

testData = trainData(:,4:5);
resultCluster = testART1(testData, W, T, r);
disp('result cluster for test data');
disp(resultCluster)

