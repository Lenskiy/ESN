%Find most correlated pairs of predicted and true signals
%Return lag for such signals
function [best_cor, best_lag, CorMat, LagMat] = esn_cor_best_match(Yt, Yp)
    Ndim = size(Yt, 2);
    CorMat = zeros(Ndim, Ndim);
    LagMat = zeros(Ndim, Ndim);
    
    for i = 1:Ndim
        for j = 1:Ndim
            [acor,lag] = xcorr(Yt(:, i), Yp(:, j), 'coeff');
            [~,I] = max(abs(acor));
            timeDiff = lag(I);
            CorMat(i, j) = acor(I);
            LagMat(i, j) = timeDiff;
        end
    end
    
    best_cor = max(max(CorMat));
    [a, b] = find(CorMat == best_cor);
    best_lag = LagMat(a, b);
%     best_cor = zeros(1, Ndim);
%     best_lag = zeros(1, Ndim);
%     tCorMat = CorMat;
%     for i = 1:Ndim
%         m = max(max(tCorMat));
%         [a, b] = find(tCorMat == m);
%         best_cor(a) = b;
%         best_lag(a) = LagMat(a, b);
%         tCorMat(a, :) = -inf;
%         tCorMat(:, b) = -inf;
%     end
end