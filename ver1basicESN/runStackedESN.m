
function Y = runStackedESN(initial_vector, Nsamples,  x, Win, W, Wout, lr, sigma)
    % in generative mode no need to initialize the states i.e. x, 
    % they have already been initialized
    % only one initial input vector is need to start generating the output
    outSize = size(Wout,1);
    Y = zeros(Nsamples, outSize);
    u = initial_vector';
    
    for j = 1:Nsamples
        x_total = [1;u];
        for  k = 1:length(W)
            x{k} = (1 - lr(k)) * x{k} + lr(k) * tanh( Win{k} * [1; u] + W{k} * x{k}) + sigma * randn(size(x{k},1), 1);
            %Relu
            %x{k} = (1 - lr(k)) * x{k} + lr(k) * max( Win{k} * [1; u] + W * x{k}, 0);
            %logistic
            %x{k} = (1 - lr(k)) * x{k} + lr(k) * tanh( Win{k} * [1; u] + W * x{k})/2 + 0.5; %
            x_total = [x_total; x{k}];
        end
        y = Wout*x_total;
        Y(j,:) = y;
        u = 1.0 * y; %generative mode
    end
end


