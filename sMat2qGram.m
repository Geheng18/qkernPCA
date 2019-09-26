function G = sMat2qGram(sMatrix,varargin)
%   sMat2qGram changes the similar-matrix of data points X to the
%   Gram-matrix using a kernel function
%   G = sMat2qGram(simMatrix, kernel,varargin)
% Inputs:
%   sMatrix: pair-wise distance matrix
%
% Output:
%   G: Gram-matrix with a conditionally negative definite kernel
%
% Yusen Zhang (yusenzhang@126.com), 2017/7/13
if nargin > 1
    kernel = varargin{1};
    if length(varargin) > 1 & strcmp(class(varargin{2}), 'double'),
        param = varargin{2};
    end
    if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'),
        sigma = varargin{3};
    end
end
G = qgram(sMatrix,kernel,param,sigma);
G=max(G,G');
end
