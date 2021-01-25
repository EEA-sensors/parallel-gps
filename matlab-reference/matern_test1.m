%
% Test Matern state-space forms by plotting the covariance functions
%

    magnSigma2 = 2;
    lengthScale = 1;

    matern_k = @(t,nu) magnSigma2.* ...
        1/gamma(nu)*2^(1-nu)*(sqrt(2*nu)*abs(t)/lengthScale).^nu.* ...
        besselk(nu,sqrt(2*nu)*abs(t)/lengthScale);

    clf;
    subplot(1,2,1);

    [Pinf,F,L,H,q] = matern32_to_ss(magnSigma2,lengthScale);
    
    t = -2:0.1:2;
    h = plot(t,ss_cov(t,F,L,q,H),t,matern_k(t,3/2),'--');
    set(h,'LineWidth',2);
    title('C(\tau)');

    [Pinf,F,L,H,q] = matern52_to_ss(magnSigma2,lengthScale);
    
    subplot(1,2,2);
    h = plot(t,ss_cov(t,F,L,q,H),t,matern_k(t,5/2),'--');
    set(h,'LineWidth',2);
    title('C(\tau)');
    