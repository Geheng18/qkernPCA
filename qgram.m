function G = qgram(X,kernel, paramq, paramp)
% GRAM Computes the Gram-matrix of data points X using a kernel function
% G = pqgram(X, X, kernel, paramp, paramq, paramc)
% Computes the Gram-matrix of data points X and X using the specified kernel
% function. If no kernel is specified, no kernel function is applied. The
% function GRAM is than equal to X*X'. The use of the function is different
% depending on the specified kernel function (because different kernel
% functions require different parameters. The possibilities are listed below.
% Linear kernel: G = gram(X, X, 'linear')
% which is parameterless
% Gaussian kernel: G = gram(X, X, 'gauss', s)
% where s is the variance of the used Gaussian function (default = 1).
% Polynomial kernel: G = gram(X, X, 'poly', R, d)
% where R is the addition value and d the power number (default = 0 and 3)
% add new
% q-base kernel: G = gram(X, X, 'qbase', q)
% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you
% maintain the name of the original author.

%(C) Laurens van der Maaten, Delft University of Technology

% Rewritten by Yusen Zhang (yusenzhang@126.com) 2017/5/18
[row,sigma]=size(X);
switch kernel
    %Non Linear Kernel
    case 'nonlbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        alpha=paramp;
        G0 = repmat(diag(X*X'),1,row);
        G1 = q.^(-alpha*G0);
        G2 = G0';
        G2 = q.^(-alpha*G2);
        G3 = X*X';
        G3 = 2*q.^(-alpha*G3);
        G = G1+G2-G3;
        G = G/(2*(1-q));
        %  Gaussian Kernel
    case 'rbfbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        Gamma=paramp;
        G = L2_distance(X', X');
        G = q.^(G.^2/Gamma);
        G = (1-G)/(1-q);
        % Laplacian Kernel
    case 'laplbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        Gamma=paramp;
        G =L2_distance(X', X');
        G = q.^(G/Gamma);
        G = (1-G)/(1-q);
        % Rational Quadratic Kernel
    case 'ratibase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        c=paramp;
        G =L2_distance(X', X');
        G = q.^(G.^2/(G.^2+c));
        G = (1-G)/(1-q);
        % Multiquadric Kernel
    case 'multbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        c=paramp;
        G =L2_distance(X', X');
        G = q.^sqrt(G.^2+c.^2);
        G = (q^c-G)/(1-q);
        %Inverse Multiquadric Kernel
    case 'invbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        c=paramp;
        G =L2_distance(X', X');
        G = q.^(-1./sqrt(G.^2+c.^2));
        G = (q^(-1/c)-G)/(1-q);
        %Wave Kernel
    case 'wavbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 0.5; end
        q=paramq;
        Theta=paramp;
        G =L2_distance(X', X');
        G = q.^(-Theta.*sin(G/Theta)./G);
        G = (q^(-1)-G)/(1-q);
        %Power Kernel
    case 'powbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        d=paramp;
        G =L2_distance(X', X');
        G = q.^(G.^d);
        G = (1-G)/(1-q);
        % Log Kernel
    case 'logbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        d=paramp;
        G =L2_distance(X', X');
        G = q.^log((G.^d)+1);
        G = (1-G)/(1-q);
        %Cauchy Kernel
    case 'caubase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end    %0< alpha <= 2
        q=paramq;
        Gamma=paramp;
        G =L2_distance(X', X');
        G = q.^(-1./(1+G.^2/Gamma));
        G = (q^(-1)-G)/(1-q);
        %Chi-Square Kernel
    case 'chibase'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        q=paramq;
        G = zeros(row);
        for i = 1:row
            for j = i+1:row
                G(i,j) = sum(2*((X(i,:)-X(j,:)).^2)./(X(i,:)+X(j,:)));
            end
        end
        G = G+G';
        G = (1-q.^G)/(1-q);
        %Generalized T-Student Kernel
    case 'studbase'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        q=paramq;
        d=paramp;
        G =L2_distance(X', X');
        G = q.^(-1./(1+G.^d));
        G = (q^(-1)-G)/(1-q);
    otherwise
        error('Unknown kernel function.');
end