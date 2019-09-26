function G = cndgram(X,kernel, paramq, paramp)
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
    %Linear Kernel
    case 'lbase'
        G = L2_distance(X', X');
        G = G.^2/2;
        %Non Linear Kernel
    case 'nonlcnd'
        if ~exist('paramq', 'var'),paramq = 1.1; end
        alpha=paramq;
        G0 = repmat(diag(X*X'),1,row);
        G1 = exp(alpha*G0);
        G2 = G0';
        G2 = exp(alpha*G2);
        G3 = X*X';
        G3 = exp(alpha*G3);
        G = G1+G2-G3;
        G = G/2;
        %Polynomial Kernel
    case 'polycnd'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        alpha=paramq;
        c=paramp;
        d=1;
        G0 = repmat(diag(X*X'),1,row);
        G1 = (alpha*G0+c).^d;
        G2 = G0';
        G2 = (alpha*G2+c).^d;
        G3 = X*X';
        G3 = (alpha*G3+c).^d;
        G = G1+G2-G3;
        G = G/2;
        %Gaussian kernel
    case 'rbfcnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        Gamma=paramq;
        G = L2_distance(X', X');
        G = exp(-G.^2/Gamma);
        G = 1-G;
        %Laplacian Kernel
    case 'laplcnd'
        if ~exist('paramq', 'var'),paramq = 0.1;  end
        Gamma=paramq;
        G = L2_distance(X', X');
        G = exp(-G/Gamma);
        G = 1-G;
        %ANOVA Kernel
    case 'anocnd'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        Delte=paramq;
        d=paramp;
        G = zeros(row);
        for i = 1:row
            for j = i+1:row
                temp = [];
                for k = 1:sigma
                    temp = [temp,(exp(-Delte*(X(i,:).^k-X(j,:).^k).^2)).^d];
                end
                G(i,j) = sum(temp);
            end
        end
        G = G+G';
        G = row-G;
        %Rational Quadratic Kernel
    case 'raticnd'
        if ~exist('paramq', 'var'),paramq = 0.1;  end
        c=paramq;
        G = L2_distance(X', X');
        G = (G.^2)./(G.^2+c);
        %Multiquadric Kernel
    case 'multcnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        c=paramp;
        G = L2_distance(X', X');
        G = sqrt(G.^2+c^2)-c;
        %Inverse Multiquadric Kernel
    case 'invcnd'
        if ~exist('paramq', 'var'),paramq = 0.1;end
        c=paramq;
        G = L2_distance(X', X');
        G = sqrt(G.^2+c^2);
        G = 1/c-1./G;
        %Wave Kernel
    case 'wavcnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        Theta=paramq;
        G = L2_distance(X', X');
        G = -Theta*sin(G/Theta)./G;
        G = 1-G;
        %Power Kernel
    case 'powcnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        d=paramq;
        G = L2_distance(X', X');
        G = G.^d;
        %Log Kernel
    case 'logcnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        d=paramq;
        G = L2_distance(X', X');
        G = log(G.^d+1);
        %Bessel Kernel
    case 'besscnd'
        if ~exist('paramp', 'var'),paramq = 0.1; paramp = 1.1; end
        Delte=paramq;
        v=paramp;
        G = L2_distance(X', X');
        G = (J(Delte*G))./(G.^(-sigma*(v+1)));
        G = 1-G;
        %Cauchy Kernel
    case 'caucnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        Gamma=paramq;
        G =L2_distance(X', X');
        G = 1./(1+G.^2/Gamma);
        G = 1-G;
        %Chi-Square Kernel %这个暂时留下
    case 'chicnd'
        G = zeros(row);
        for i = 1:row
            for j = i+1:row
                G(i,j) = sum(2*((X(i,:)-X(j,:)).^2)./(X(i,:)+X(j,:)));
            end
        end
        G = G+G';
        %Generalized T-Student Kernel
    case 'studcnd'
        if ~exist('paramq', 'var'),paramq = 0.1; end
        d=paramq;
        G =L2_distance(X', X');
        G = 1./(1+G.^d);
        G = 1-G;
    otherwise
        error('Unknown kernel function.');
end