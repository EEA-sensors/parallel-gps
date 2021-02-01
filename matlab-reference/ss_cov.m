% cov_approx = ss_cov(tau,F,L,q,H,Pinf)
%
% Covariance function of state space model.

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

% Copyright (C) 2014 Simo Sarkka

function cov_approx = ss_cov(tau,F,L,q,H,Pinf)

    if nargin < 6
        Pinf = lyapchol(F,L*sqrt(q));
        Pinf = Pinf' * Pinf;
    end

    % Initialize covariance
    cov_approx = zeros(size(tau));
  
    % Evaluate positive parts
    cov_approx(tau >= 0) = arrayfun(@(taut) H*Pinf*expm(taut*F)'*H',tau(tau >= 0));
  
    % Evaluate negative parts
    cov_approx(tau < 0) = arrayfun(@(taut) H*expm(-taut*F)*Pinf*H',tau(tau < 0));
end

