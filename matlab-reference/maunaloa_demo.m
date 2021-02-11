%
% Run Mauna Loa model with fixed parameters with KFS. Uses borrowed
% codes from Solin/Särkkä:2014

    %%
    % Load and preprocess data
    %
    
    % Load and convert
    monthly = textread('co2_mm_mlo.txt','','commentstyle','shell');
    weekly  = textread('co2_weekly_mlo.txt','','commentstyle','shell');
    
    % Fix missing values
    monthly(monthly<-99) = nan;
    weekly(weekly<-99)  = nan;
    
    
    % Combine monthly and weekly data
    
    % First weekly value date
    t0 = min(weekly(:,4));
    
    % Indices in monthly values with dates smaller than t0
    ind = (monthly(:,3) < t0);
    
    % Combine monthly and weekly data
    t = [monthly(ind,3); weekly(:,4)];
    y = [monthly(ind,4); weekly(:,5)];
    
    % Remove nans (missing values)
    ind = ~isnan(y);
    t   = t(ind);
    y   = y(ind);
    
    % Only use data prior to 2010, retain newer for validation
    ind   = t<2010;
    yn    = y(ind);
    ymean = mean(yn);
    yn    = yn-ymean;
    to    = t(ind);
    xt    = (2010:1/24:2020)';
    
    % Show
    
    figure(1); clf
    plot(to,yn+ymean,'xk','MarkerSize',3)
    xlabel('Time (year)'); ylabel('Observed CO_2 concentration (ppm)')
    axis tight

    %%
    % Form the model. In reality we should optimize all magSigma2 and
    % lengthScale parameters but we don't now have derivatives.
    %
    se_lengthScale = 100; 
    se_magnSigma2  = 1e4;
    ma_lengthScale = 1;
    ma_magnSigma2  = 0.5;
    qp_lengthScale = 1;
    qp_magnSigma2  = 5;
    period         = 1;
    mlengthScale   = 140
    damping        = 'matern32';
    
    [Pinf,F,L,H,q,H_se,H_ma,H_qp] = maunaloa_to_ss(se_magnSigma2, se_lengthScale, ...
        ma_magnSigma2, ma_lengthScale, ...
        qp_magnSigma2, qp_lengthScale, ...
        period, mlengthScale, damping);

    % Run Kalman filter
    m = zeros(size(Pinf,1),1);
    P = Pinf;
    R = 1;

    all_t = [to;xt];

    pred_m = zeros(1,length(all_t));
    pred_v = zeros(1,length(all_t));

    for k=1:length(all_t)
        if k > 1
            dt = all_t(k) - all_t(k-1);
            [A,Q] = lti_disc(F,L,q,dt);
            m = A*m;
            P = A*P*A' + Q;
        end

        if k <= length(yn)
            S = H * P * H' + R;
            K = P * H' / S;
            m = m + K * (yn(k) - H * m);
            P = P - K * S * K';
        end

        pred_m(k) = H * m;
        pred_v(k) = H * P * H';


    end

    
    clf;
    h = plot(all_t,ymean + pred_m)
    set(h,'MarkerSize',3,'LineWidth',0.1);
    
    hold on;
    
    % Show the observations
    h = plot(t,y,'xk');
    set(h,'MarkerSize',3,'LineWidth',0.1);
    
    