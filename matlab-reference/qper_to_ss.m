% Form state-space representation of quasiperiodic covariance function
function [Pinf,F,L,H,q] = qper_to_ss(magnSigma2, lengthScale, period, mlengthScale, damping, N)
    if nargin < 6
        N = 6;
    end
    
    % From codes of Solin/Särkkä:2014
    [Pinf1,F1,L1,H1,q1] = per_to_ss(magnSigma2,lengthScale,period,N);     
    [Pinf2,F2,L2,H2,q2] = feval(sprintf('%s_to_ss',damping),1,mlengthScale);
    
    [Pinf,F,L,H,q] = prod_cov(Pinf1,F1,L1,H1,q1,Pinf2,F2,L2,H2,q2);
end
