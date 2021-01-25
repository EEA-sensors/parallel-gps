% Form state-space representation of Mauna Loa CO2 model
function [Pinf,F,L,H,q,H_se,H_ma,H_qp] = maunaloa_to_ss(se_magnSigma2, se_lengthScale, ...
                                         ma_magnSigma2, ma_lengthScale, ...
                                         qp_magnSigma2, qp_lengthScale, ...
                                         period, mlengthScale, damping)

    % Based on codes of Solin/Särkkä:2014
    [Pinf1,F1,L1,H1,q1] = se_to_ss(se_magnSigma2,se_lengthScale);
    [Pinf2,F2,L2,H2,q2] = matern32_to_ss(ma_magnSigma2,ma_lengthScale);
    [Pinf3,F3,L3,H3,q3] = qper_to_ss(qp_magnSigma2, qp_lengthScale, period, ...
                                     mlengthScale, damping);

    % Stack
    F    = blkdiag(F1,F2,F3);
    L    = blkdiag(L1,L2,L3);
    q    = blkdiag(q1,q2,q3);
    H    = [H1 H2 H3];
    Pinf = blkdiag(Pinf1,Pinf2,Pinf3);
    
    H_se = [H1 zeros(size(H2)) zeros(size(H3))];
    H_ma = [zeros(size(H1)) H2 zeros(size(H3))];
    H_qp = [zeros(size(H1)) zeros(size(H2)) H3];
end

