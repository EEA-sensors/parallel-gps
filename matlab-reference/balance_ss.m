% Balance state-space model to that it is numerically more stable
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

