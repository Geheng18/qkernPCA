function [mappedX,evecs,mapping] = qKPCA(X,no_dims,varargin)
%KERNEL_PCA Perform the kernel PCA algorithm
%input£º
%      X : matrix: row is sample and column is feature
%      no_dims : reduce to no_dims
%      gram : switch qgram or cndgram to compute
%output£º
%      mappedX : no_dims matrix
% The function runs kernel PCA on a set of datapoints X. The variable
% no_dims sets the number of dimensions of the feature points in the
% embedded feature space (no_dims >= 1, default = 2).
% For no_dims, you can also specify a number between 0 and 1, determining
% the amount of variance you want to retain in the PCA step.
% The value of kernel determines the used kernel. Possible values are 'linear',
% 'gauss', 'poly', 'qbase', or 'princ_angles' (default = 'gauss'). For
% more info on setting the parameters of the kernel function, type HELP
% GRAM.
% The function returns the locations of the embedded trainingdata in
% mappedX. Furthermore, it returns information on the mapping in mapping.
% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
%
% Modified by Yusen Zhang (yusenzhang@126.com) 11 may 2017;
%% set the parameter
if ~exist('no_dims', 'var')
    no_dims = 2;
end
[train_num,nCol] = size(X);
gram = 'qgram';
kernel = 'laplbase';
q = 0.9;
sigma = nCol;
if nargin > 2
    if length(varargin) > 0 & strcmp(class(varargin{1}), 'char'),
        gram = varargin{1};
    end
    if length(varargin) > 1 & strcmp(class(varargin{2}), 'char'),
        kernel = varargin{2};
    end
    if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'),
        q = varargin{3};
    end
    if length(varargin) > 3 & strcmp(class(varargin{4}), 'double'),
        sigma = varargin{4};
    end
end
%% compute the kernel gram matrix 
%data normalize
mX_train = mean(X);
Z_train = X - repmat(mX_train,train_num,1);
disp('Using method qKPCA');
disp(['Using ',kernel,' with q = ',num2str(q),' and sigma = ',num2str(sigma)]);
disp('Computing kernel matrix...');
% compute the simMat
sMatrix = sqrt(bsxfun(@plus,sum(Z_train.^2,2),bsxfun(@plus,sum(Z_train.^2,2)',-2*(Z_train*Z_train'))));   
sMatrix = real(sMatrix);
switch gram
    case 'qgram'
        H = sMat2qGram(sMatrix,kernel,q,sigma);
    case 'cndgram'
        H = sMat2cndGram(sMatrix,kernel,q,sigma);
    otherwise
        disp("please check you parameter on switching kernels");
end
H = max(H,H');
%% compute the first no_dims components
% Normalize kernel matrix K
unit = ones(train_num, 1);
K_center = H - unit*sum(H)/train_num - unit*sum(H)/train_num' + sum(sum(H)/train_num)/train_num;
K_center = -K_center;
K_center = max(K_center,K_center');
% Compute first no_dims eigenvectors and store these in V, store corresponding eigenvalues in L
disp('Eigenanalysis of kernel matrix...');
K_center(isnan(K_center)) = 0;
K_center(isinf(K_center)) = 0;
[evecs, evals] = eig(K_center);
% Sort eigenvalues and eigenvectors in descending order
dims = min(train_num,no_dims);
[evals, ind] = sort(diag(evals), 'descend');
evals = evals(1:dims);
evecs = evecs(:,ind(1:dims));
% Compute inverse of eigenvalues matrix L
disp('Computing final embedding...');
invL = diag(1 ./ evals);
% Compute square root of eigenvalues matrix L
sqrtL = diag(sqrt(evals));
% Compute inverse of square root of eigenvalues matrix L
invsqrtL = diag(1 ./ diag(sqrtL));
% Compute the new embedded points for both K and Ktest-data
mappedX = sqrtL * evecs';            
% Set feature vectors in original format
mappedX = mappedX';
% Store information for out-of-sample extension
mapping.X = X;
mapping.kernel = kernel;
mapping.q = q;
mapping.sigma = sigma;
end
