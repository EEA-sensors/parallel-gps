%
% Test quasiperiodic covariance functions by plotting covariances
%

    magnSigma2 = 2;
    lengthScale = 1;
    period = 0.5;
    mlengthScale = 1;

    qper_k = @(t,nu) magnSigma2.* ...
        exp(-2*sin(2*pi/period*t/2).^2/lengthScale^2).* ...
        1/gamma(nu).*2^(1-nu).*(sqrt(2*nu)*abs(t)/mlengthScale).^nu.* ...
        besselk(nu,sqrt(2*nu)*abs(t)/mlengthScale);

    damping = 'se';
    [Pinf,F,L,H,q] = qper_to_ss(magnSigma2, lengthScale, period, mlengthScale, damping);
    
    clf;
    subplot(1,2,1);
    
    t = -2:0.01:2;
    h = plot(t,ss_cov(t,F,L,q,H),t,qper_k(t,100),'--');
    set(h,'LineWidth',2);
    title('C(\tau)');

    damping = 'matern32'; nu = 3/2;
%    damping = 'matern52'; nu = 5/2;
    [Pinf,F,L,H,q] = qper_to_ss(magnSigma2, lengthScale, period, mlengthScale, damping);

    subplot(1,2,2);
    
    t = -2:0.01:2;
    h = plot(t,ss_cov(t,F,L,q,H),t,qper_k(t,nu),'--');
    set(h,'LineWidth',2);
    title('C(\tau)');
    