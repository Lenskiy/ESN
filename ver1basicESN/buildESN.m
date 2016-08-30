% input_size defines the number of inputs
% reservoir_size defines the number of neurons in the reservoir
% connectivity is the portion of the non-zero connections in the reservoir
% sp is the spectral radius that W will have 
% Win is the matrix of input weights
% W defines weights between neurons in the reservoir
function [Win, W] = buildESN(input_size, reservoir_size, connectivity, sp)
    inputScaling = 1;

    rng('shuffle');
    % generate input weigh matrix randomly
    Win = (rand(reservoir_size,1+input_size) - 0.5) .* inputScaling;
    
    % generate connecting weight in reservoir
    W = sprand(reservoir_size, reservoir_size, connectivity);
    W(W ~= 0) = W(W ~= 0) - 0.5;
   
    % compute spectral radius i.e. the largest absolute eigen value 
	opts.tol = 1e-3;
    maxVal = max(abs(eigs(W, 1, 'lm', opts)));
	W = sp * W/maxVal; % normalize W such that spectral radius is sp
 end