% Convert squared exponential covariance function to state-space form
function [Pinf,F,L,H,q] = se_to_ss(magnSigma2,lengthScale,order)
    
    if nargin < 3
        order = 6;
    end

    %
    % ------ NOTE: This part can be done in CPU:
    %

    %
    % Form the canonical state-space representation of SE
    %
    
    % First form the inverse Taylor series for the Gaussian kernel
    
    B = sqrt(2*pi);
    A = zeros(1,2*order+1);

    i = 1;
    for k=order:-1:0
        A(i) = 0.5^k/factorial(k);
        i = i + 2;
    end

    % Convert to state space form -- this is a general routine actually
    q = polyval(B,0) / polyval(A,0);
    
    LB = B ./ 1i.^(length(B)-1:-1:0);
    LA = A ./ 1i.^(length(A)-1:-1:0);

    BR = roots(LB);
    AR = roots(LA);
    
    GB = poly(BR(real(BR) < 0));
    GA = poly(AR(real(AR) < 0));
    
    GB = GB ./ GB(end);
    GA = GA ./ GA(end);
    
    GB = GB ./ GA(1);
    GA = GA ./ GA(1);

    F = zeros(length(GA)-1);
    F(end,:) = -GA(end:-1:2);
    F(1:end-1,2:end) = eye(length(GA)-2);
    
    L = zeros(length(GA)-1,1);
    L(end) = 1;
    
    H = zeros(1,length(GA)-1);
    H(1:length(GB)) = GB(end:-1:1);
    
    % -> F,L,G,Q should go to GPU now
    
    %
    % ---- NOTE: This part should be done in GPU:
    %
    
    % Rescale for non-unity s and ell
    dim = size(F,1);
    ell_vec = lengthScale.^(dim:-1:1);
    F(end,:) = F(end,:) ./ ell_vec;
    H(1) = H(1) / lengthScale^dim;
    q = magnSigma2 * lengthScale * q;

    % Balance
    [F,L,q,H] = balance_ss(F,L,q,H);
    
    % Solve the Lyapunov
    Pinf = solve_lyap(F,L,q);

end
