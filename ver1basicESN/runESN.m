
function Y = runESN(initial_vector, Nsamples,  states, Win, W, Wout, lr, sigma)
    % in generative mode no need to initialize the states, 
    % they have already been initialized
    % only one initial input vector is need to start generating the output
    outSize = size(Wout,1);
    Y = zeros(Nsamples, outSize);
    u = initial_vector';
    
	for k = 1:Nsamples
        states = (1 - lr) * states + lr * tanh( Win * [1; u] + W * states) + sigma * randn(size(states,1), 1);
        y = Wout*[1;u;states];
        Y(k,:) = y;
        u = 1.0 * y; %generative mode
	end
end


