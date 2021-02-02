function [Pinf,F,L,H,q] = prod_cov(Pinf1,F1,L1,H1,q1,Pinf2,F2,L2,H2,q2)
    F    = kron(F1,eye(size(F2))) + kron(eye(size(F1)),F2);
%    L    = kron(L1,L2);
%    q    = kron(q1,q2);
    q    = kron(L1*q1*L1',Pinf2) + kron(Pinf1,L2*q2*L2')
    L    = eye(size(q));
    Pinf = kron(Pinf1,Pinf2);
    H    = kron(H1,H2);
end
