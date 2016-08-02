% input_size defines the number of inputs
% reservoir_size defines the number of neurons in the reservoir
% connectivity is the portion of the non-zero connections in the reservoir
% sp is the spectral radius that W will have 
% Win is the matrix of input weights
% W defines weights between neurons in the reservoir
function [sWin, sW] = buildStackedESN(input_size, reservoir_size, connectivity, sp, input_weight_type)
    inputScaling = 1;

    rng('shuffle');
    for k = 1:length(reservoir_size)
        % generate input weigh matrix randomly
        switch (input_weight_type)
            case 'const'
                sWin{k} = ones(reservoir_size(k), 1 + input_size) / (reservoir_size(k) * input_size);
            case 'rand'
                sWin{k} = (rand(reservoir_size(k),1 + input_size) - 0.5) .* inputScaling;
            case 'randn'
                sWin{k} = (randn(reservoir_size(k),1 + input_size)) .* inputScaling;
        end

        % generate connecting weight in reservoir
        W = sprand(reservoir_size(k), reservoir_size(k), connectivity(k));
        W(W ~= 0) = W(W ~= 0) - 0.5;

        % compute spectral radius i.e. the largest absolute eigen value 
        opts.tol = 1e-2;
        maxVal = max(abs(eigs(W, 1, 'lm', opts)));
        W = sp(k) * W/maxVal; % normalize W such that spectral radius is sp
        sW{k} = W;
    end
    
end