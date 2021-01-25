% Form state-space representation of quasiperiodic covariance function
function [Pinf,F,L,H,q] = qper_to_ss(magnSigma2, lengthScale, period, mlengthScale, damping, N)
    if nargin < 6
        N = 6;
    end
    
    % From codes of Solin/Särkkä:2014
    [Pinf1,F1,L1,H1,q1] = per_to_ss(magnSigma2,lengthScale,period,N);     
    [Pinf2,F2,L2,H2,q2] = feval(sprintf('%s_to_ss',damping),1,mlengthScale);
    
    F    = kron(F1,eye(size(F2))) + kron(eye(size(F1)),F2);
    L    = kron(L1,L2);
    q    = kron(Pinf1,q2);
    Pinf = kron(Pinf1,Pinf2);
    H    = kron(H1,H2);
end

