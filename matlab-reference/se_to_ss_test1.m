%
% Testing routine for squared exponential -- just plot covariances
%

    s = 1;
    ell = 0.1;
    n = 6;
    
    se_spec = @(w) s^2 * sqrt(2*pi) * ell * exp(-ell^2 * w.^2/2);
    se_cov  = @(t) s^2 * exp(-t.^2/2/ell^2);

    [Pinf,F,L,H,q] = se_to_ss(s^2,ell,n);
    
    clf;
    subplot(1,2,1);
    
    w = -5:0.1:5;
    h = plot(w,ss_spec(w,F,L,q,H),w,se_spec(w),'--');
    set(h,'LineWidth',2);
    title('S(\omega)');

    subplot(1,2,2);
    
    tau = -4:0.01:4;
    h = plot(tau,ss_cov(tau,F,L,q,H),tau,se_cov(tau),'--');
    set(h,'LineWidth',2);
    title('C(\tau)');
    
    
    