function [mappedX,evecs,mapping] = JshqKPCA(X,no_dims,varargin)
% This function does principal component analysis (non-linear) on the given
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
% ambarish.jash@colorado.edu
% modified by Yusen Zhang (yusenzhang@126.com) 22 April 2017
% Checking to ensure output dimensions are lesser than input dimension.
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
Z_train = X - repmat(mX_train,train_num,1);
disp('Using method JshqKPCA');
disp('Computing kernel matrix...');
disp(['Using ',kernel,' with q = ',num2str(q),' and sigma = ',num2str(sigma)]);
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
unit = ones(size(H));
K_center = H - unit*H - H*unit + unit*H*unit;
K_center = -K_center;
K_center = max(K_center,K_center');
% Obtaining the low dimensional projection
% The following equation needs to be satisfied for K
% N*lamda*K*alpha = K*alpha
% Thus lamda's has to be normalized by the number of points
disp('Eigenanalysis of kernel matrix...');
opts.issym = 1; % opts.issym = 1 means that A is a symmetric matrix
opts.disp = 0; % degree of display 
opts.isreal = 1; % opts.isreal = 1 means that A is a real matrix
neigs = 30;
% R2017b and later version should use 'largestabs' instead of 'lm'
[evecs,evals] = eigs(K_center,[],neigs,'lm',opts); 
eigval = evals ~= 0;
eigval = eigval./nCol;
% Again 1 = lamda*(alpha.alpha)
for col = 1:size(evecs,2)
    evecs(:,col) = evecs(:,col)./(sqrt(eigval(col,col)));
end
[~, ind] = sort(eigval,'descend');
evecs = evecs(:,ind);
% Projecting the data in lower dimensions
mappedX = zeros(no_dims,size(K_center,2));
for count = 1:no_dims
    mappedX(count,:) = evecs(:,count)' * K_center';
end
mappedX = mappedX';
% Store information for out-of-sample extension
mapping.X = X;
mapping.kernel = kernel;
mapping.q = q;
mapping.sigma = sigma;
end
