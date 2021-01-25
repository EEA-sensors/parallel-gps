%
% Test the Mauna Loa model state space representation by plotting the
% covariance

    se_lengthScale = 100;
    se_magnSigma2  = 1e4;
    ma_lengthScale = 1;
    ma_magnSigma2  = 0.5;
    qp_lengthScale = 1;
    qp_magnSigma2  = 5;
    period         = 1;
    mlengthScale   = 140
    damping        = 'matern32';
    
    
    % Define covariance functions: quasi-periodic
    nu  = 3/2;
    kp = @(t) qp_magnSigma2.* ...
        exp(-2*sin(2*pi/period*t/2).^2/qp_lengthScale^2).* ...
        1/gamma(nu).*2^(1-nu).*(sqrt(2*nu)*abs(t)/mlengthScale).^nu.* ...
        besselk(nu,sqrt(2*nu)*abs(t)/mlengthScale);
    
    % Define covariance functions: Matern (nu=3/2)
    nu  = 3/2;
    k32 = @(t) ma_magnSigma2.* ...
        1/gamma(nu)*2^(1-nu)*(sqrt(2*nu)*abs(t)/ma_lengthScale).^nu.* ...
        besselk(nu,sqrt(2*nu)*abs(t)/ma_lengthScale);
    
    % Define covariance functions: Matern (nu=inf)
    kse = @(t) se_magnSigma2.*exp(-t.^2/2/se_lengthScale.^2);

    [Pinf,F,L,H,q,H_se,H_ma,H_qp] = maunaloa_to_ss(se_magnSigma2, se_lengthScale, ...
        ma_magnSigma2, ma_lengthScale, ...
        qp_magnSigma2, qp_lengthScale, ...
        period, mlengthScale, damping);
    
    
    t = -10:0.1:10;
    
    clf;
    subplot(2,2,1);    
    plot(t,ss_cov(t,F,L,q,H),t,kp(t) + k32(t) + kse(t),'--');
    
    subplot(2,2,2);    
    plot(t,ss_cov(t,F,L,q,H_se),t,kse(t),'--');

    subplot(2,2,3);    
    plot(t,ss_cov(t,F,L,q,H_ma),t,k32(t),'--');

    subplot(2,2,4);    
    plot(t,ss_cov(t,F,L,q,H_qp),t,kp(t),'--');

    
    
    