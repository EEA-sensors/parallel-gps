% Balance state-space model to that it is numerically more stable
%
%   [F,L,q,H,D] = balance_ss(F,L,q,H,iter)
%
% Finds a diagoal matrix D such that the system with state z <- D^{-1} x
% is more stable than the original one. This transform the system
%
%    dx/dt = F x + L w
%        y = H x
%
%  to
%
%    dz/dt = D^{-1} F D z + D^{-1} L w
%        y = H D z
%
% The Lyapunov equation
%
%   F Pinf + Pinf F^T + L Q L^T = 0
%
% then becomes
%
%   D^{-1} F D Pinf + Pinf D F^T D^{-1} + D^{-1} L Q L D^{-1} = 0
%   F (D Pinf D^{-1}) + (D^{-1} Pinf D) F^T + L Q L^T = 0
%
% i.e.
%
%   Pinf <- D Pinf D^{-1}
%   
function [F,L,q,H,D] = balance_ss(F,L,q,H,iter)
    if nargin < 5
        iter = 5;
    end

    %
    % Balance the state space modeling using Alg 1 in https://arxiv.org/pdf/1401.5766.pdf
    % Finally, scale the model to have norm_oo(H) = 1 and norm_oo(L) = 1
    %
    
    dim = size(F,1);
    D = eye(dim);
    for k=1:iter
        for i=1:dim
            tmp = F(:,i);
            tmp(i) = 0;
            c = norm(tmp);
            tmp = F(i,:);
            tmp(i) = 0;
            r = norm(tmp);
            f = sqrt(r/c);
            D(i,i) = f * D(i,i);
            F(:,i) = f * F(:,i);
            F(i,:) = F(i,:) / f;
            L(i,:) = L(i,:) / f;
            H(:,i) = f * H(:,i);
        end
    end
    
    tmp = max(abs(L));
    L = L ./ tmp;
    q = tmp^2 * q;
    
    tmp = max(abs(H));
    H = H ./ tmp;
    q = tmp^2 * q;

end

