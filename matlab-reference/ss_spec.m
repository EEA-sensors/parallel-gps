% S_approx = ss_spec(w,F,L,q,H)
%
% Spectral density of state space model.

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

% Copyright (C) 2014 Simo Sarkka

function S_approx = ss_spec(w,F,L,q,H)

    S_approx = arrayfun(@(wt) H/(F-1i*wt*eye(length(F)))*L*q*L'/((F+1i*wt*eye(length(F))).')*H',w);
    S_approx = real(S_approx);
end

