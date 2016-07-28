
function Y = runStackedESN(initial_vector, Nsamples,  x, Win, W, Wout, lr, sigma, node_type, output_type, feedback_scaling )
    % in generative mode no need to initialize the states i.e. x, 
    % they have already been initialized
    % only one initial input vector is need to start generating the output
    outSize = size(Wout,1);
    Y = zeros(Nsamples, outSize);
    u = initial_vector';
    
    switch(node_type)
        case 'tanh'
            node_type = 1;
        case 'logistic'
            node_type = 2;
        case 'relu'
            node_type = 3;
        case 'linear'
            node_type = 4;
        otherwise
            node_type = 1;
    end
    
    for j = 1:Nsamples
        x_total = [1;u];
        for  k = 1:length(W)
            switch (node_type)
                case 1 %Hypebolic tangent
                x_updated =  tanh( Win{k} * [1; u] + W{k} * x{k} ) + sigma * randn(size(x{k},1), 1);
                case 2 %Relu
                x_updated =  max( Win{k} * [1; u] + W{k} * x{k}, 0) + sigma * randn(size(x{k},1), 1);
                case 3 %logistic
                x_updated =  tanh( Win{k} * [1; u] + W{k} * x{k})/2 + 0.5 + sigma * randn(size(x{k},1), 1); 
                case 4 %linear
                x_updated =  (Win{k} * [1; u] + W{k} * x{k}) + sigma * randn(size(x{k},1), 1);
            end
            
            x{k} = (1 - lr(k)) * x{k} + lr(k) * x_updated;
            x_total = [x_total; x{k}];
        end
        
        switch (output_type)
            case 'linear'
                
            case 'quadratic'
                X = nchoosek(x_total(:,1),2);
                Y_temp = X(:,1).*X(:,2);
        
                x_total = [x_total; Y_temp];
        end
        
        y = Wout*x_total;
        Y(j,:) = y;
        u = feedback_scaling * y; %generative mode, pass outpus as an input;
    end
    
    
end


