%
% Testing of product of covariance functions
%

    %%
    % Form the individual covariance function
    %

    % Define covariance functions: Matern (nu=3/2)
    matcov = @(t,nu,magnSigma2,lengthScale) magnSigma2.* ...
        1/gamma(nu)*2^(1-nu)*(sqrt(2*nu)*abs(t)/lengthScale).^nu.* ...
        besselk(nu,sqrt(2*nu)*abs(t)/lengthScale);


    magnSigma2_1 = 0.2;
    lengthScale_1 = 1;
    [Pinf1,F1,L1,H1,q1] = matern32_to_ss(magnSigma2_1,lengthScale_1)
    
    magnSigma2_2 = 0.5;
    lengthScale_2 = 2;
    [Pinf2,F2,L2,H2,q2] = matern52_to_ss(magnSigma2_2,lengthScale_2)
    
    t = -10:0.1:10;
    
    clf;
    subplot(2,1,1);    
    h = plot(t,ss_cov(t,F1,L1,q1,H1),t,matcov(t,3/2,magnSigma2_1,lengthScale_1),'--');
    set(h,'LineWidth',2)

    subplot(2,1,2);    
    h = plot(t,ss_cov(t,F2,L2,q2,H2),t,matcov(t,5/2,magnSigma2_2,lengthScale_2),'--');
    set(h,'LineWidth',2)
    
    %%
    % Form the product and check that it matches
    %
    [Pinf,F,L,H,q] = prod_cov(Pinf1,F1,L1,H1,q1,Pinf2,F2,L2,H2,q2);

    clf;
    y = matcov(t,3/2,magnSigma2_1,lengthScale_1) .* matcov(t,5/2,magnSigma2_2,lengthScale_2);
    h = plot(t,ss_cov(t,F,L,q,H),t,y,'--');
    set(h,'LineWidth',2)
    
    %%
    % Another test
    %
    magnSigma2_1 = 1;
    lengthScale_1 = 1;
    [Pinf1,F1,L1,H1,q1] = matern52_to_ss(magnSigma2_1,lengthScale_1)
    
    s = 1;
    ell = 0.1;
    n = 6;
    
    [Pinf2,F2,L2,H2,q2] = se_to_ss(s^2,ell,n);
    
    [Pinf,F,L,H,q] = prod_cov(Pinf1,F1,L1,H1,q1,Pinf2,F2,L2,H2,q2);
    
    clf;
    t = -2:0.01:2;
    y = ss_cov(t,F1,L1,q1,H1) .* ss_cov(t,F2,L2,q2,H2);
    h = plot(t,ss_cov(t,F,L,q,H,Pinf),t,y,'--');
    set(h,'LineWidth',2)
    
    