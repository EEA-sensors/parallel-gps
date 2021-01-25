%% Calculate the coefficients q_j^2 for the approximation
function a = seriescoeff(m,lengthScale,magnSigma2)
  
    % From codes of Solin/Särkkä:2014
    % Set up the coefficients for cos(t)^k in terms of \sum_j b_j cos(j*t)
    b = @(k,j) 2.*choose(k,floor((k-j)/2).*(j<=k)) ./ ...
             (1+(j==0)*1) .* (j<=k) .* (mod(k-j,2)==0);

    % Set up mesh of indices
    [J,K] = meshgrid(0:m,0:m);

    % Calculate the coefficients
    a = b(K,J)                .* ...
        lengthScale.^(-2*K)   .* ...
        1./factorial(K)       .* ...
        exp(-1/lengthScale^2) .* ...
        2.^-K                 .* ...
        magnSigma2;

    % Sum over the Js
    a = sum(a,1);  
end

