clear
a1 = sym('a1', 'real');
a2 = sym('a2', 'real');
a3 = sym('a3', 'real');

assume(-a1, 'positive');
assume(-a2, 'positive');
assume(-a3, 'positive');

A = [0 1 0; 0 0 1; a1 a2 a3];

lam1 = sym('lam1', 'real');
lam2 = sym('lam2', 'real');
lam3 = sym('lam3', 'real');

assume(lam1, 'positive');
assume(lam2, 'positive');
assume(lam3, 'positive');

D = diag([lam1 lam2 lam3]);

% eq1 = (lam1/lam3*a1)^2 - (lam2/lam1)^2 == 0;
% eq2 = (lam3/lam2)^2 - (lam1/lam3*a1)^2 - (lam2/lam3*a1)^2 == 0;
% eq3 = (lam2/lam1)^2 + (lam2/lam3*a2)^2 - (lam3/lam2)^2 == 0;

Dinv = diag(1./diag(D));

balance = Dinv * A * D;

eq1 = norm(balance(1, :))^2 == norm(balance(:, 1))^2;
eq2 = norm(balance(2, :))^2 == norm(balance(:, 2))^2;
eq3 = norm(balance(3, :))^2 == norm(balance(:, 3))^2;

S = solve([eq1 eq2 eq3], [lam1 lam2 lam3], 'ReturnConditions',true);