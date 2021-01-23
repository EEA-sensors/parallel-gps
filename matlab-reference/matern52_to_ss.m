% Form state-space representatio of Matern 5/2
function [Pinf,F,L,H,q] = matern52_to_ss(magnSigma2,lengthScale)

    % From codes of Solin/Särkkä:2014
    lambda = sqrt(5)/lengthScale;
    F = [ 0,          1,          0;
        0,          0,          1;
        -lambda^3, -3*lambda^2, -3*lambda];
    L = [0; 0; 1];
    q = magnSigma2*400*sqrt(5)/3/lengthScale^5;
    H = [1, 0, 0];

    kappa = 5/3*magnSigma2/lengthScale^2;

    Pinf = [magnSigma2, 0,      -kappa;
        0,          kappa,  0;
        -kappa,     0,      25*magnSigma2/lengthScale^4]; 
end

