% Form state-space representatio of Matern 3/2
function [Pinf,F,L,H,q] = matern32_to_ss(magnSigma2,lengthScale)

    % From codes of Solin/Särkkä:2014

    lambda = sqrt(3)/lengthScale;
    F = [0,          1;
        -lambda^2,  -2*lambda];
    L = [0;   1];
    q = 12*sqrt(3)/lengthScale^3*magnSigma2;
    H = [1,   0];
    Pinf = [magnSigma2, 0;
        0,          3*magnSigma2/lengthScale^2];

end

