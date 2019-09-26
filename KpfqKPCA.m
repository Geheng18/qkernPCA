function [mappedX,evecs,mapping] = KpfqKPCA(X,no_dims,varargin)
% qKernel PCA
%input£º
%      X : matrix: row is sample and column is feature
%      no_dims : reduce to no_dims
%      gram : switch qgram or cndgram to compute
%output£º
%      mappedX : no_dims matrix
%   ref:
%   @article{SchSmoMue98,
%   author    = "B.~{Sch\"olkopf} and A.~Smola and K.-R.~{M\"uller}",
%   title     = "Nonlinear component analysis as a kernel Eigenvalue problem",
%   journal   =	{Neural Computation},
%   volume    = 10,
%   issue     = 5,
%   pages     = "1299 -- 1319",
%   year      = 1998}
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
% This origin file can be downloaded from http://www.kernel-machines.org.
% Last modified: 4 July 2003
% Modified by Yusen Zhang (yusenzhang@126.com) 2017/4/22
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
mX_train = mean(X);
Z_train = X-repmat(mX_train,train_num,1);
disp('Using method KpfqKPCA');
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
% We know that for PCA the data has to be centered. Even if the input data
% set 'X' lets say in centered, there is no gurantee the data when mapped
% in the feature space [phi(x)] is also centered. Since we actually never
% work in the feature space we cannot center the data. To include this
% correction a pseudo centering is done using the Kernel.
dims = min(train_num,no_dims);
unit = ones(train_num,train_num)/train_num;
% centering in feature space
K_center = H - unit*H - H*unit + unit*H*unit;
K_center = -K_center;
K_center = max(K_center,K_center');
disp('Eigenanalysis of kernel matrix...');
K_center(isnan(K_center)) = 0;
K_center(isinf(K_center)) = 0;
[evecs, evals] = eig(K_center);
% Sort eigenvalues and eigenvectors in descending order
[evals, ind] = sort(diag(evals), 'descend');
evals = evals(1:dims);
evecs = evecs(:,ind(1:dims));
% normalizing eigenvector
for i = 1:length(evals)
    evecs(:,i) = evecs(:,i)/sqrt(evals(i));
end
% extract features
mappedX = K_center * evecs(:,1:dims);
% Store information for out-of-sample extension
mapping.X = X;
mapping.kernel = kernel;
mapping.q = q;
mapping.sigma = sigma;
end
