% Form state-space representation of periodic covariance function
function [Pinf,F,L,H,q] = per_to_ss(magnSigma2, lengthScale, period, N)
    if nargin < 4
        N = 6;
    end

    % From codes of Solin/Särkkä:2014
    % The series coefficients
    q2 = seriescoeff(N,lengthScale,magnSigma2);
    
    % The angular frequency
    w0   = 2*pi/period;
    
    % The model
    F    = kron(diag(0:N),[0 -w0; w0 0]);
    L    = eye(2*(N+1));
    q    = zeros(2*(N+1));
    Pinf = kron(diag(q2),eye(2));
    H    = kron(ones(1,N+1),[1 0]);
end

