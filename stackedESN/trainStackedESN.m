
function [Wout, x, x_evolution_org] = trainStackedESN(input,...
                                                        target, Win, W, lr, sigma, node_type, output_type)
    initLen = 0;
    trainLen = size(input,1);
    input_size = size(input, 2);
    states_size = 0;
    
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
    
    % allocated memory for the design (collected states) matrix
    for k = 1:length(W)
        x{k} = zeros(size(W{k},1),1);
        states_size = states_size + size(W{k},1);
    end    

    x_evolution = zeros(1 + input_size + states_size,...
                                                      trainLen - initLen);
    % run the reservoir with the data and collect states

    
    for j = 1:trainLen
        u = input(j, :)';
        cur_state_ind = 1;
        x_evolution(cur_state_ind:input_size + 1,j - initLen) = [1; u];
        cur_state_ind = cur_state_ind + input_size + 1;
        for k = 1:length(W)
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
            
            if j > initLen % don't store a first few input states 
                x_evolution(cur_state_ind:cur_state_ind + size(x{k},1) - 1, j - initLen) = x{k};
                cur_state_ind = cur_state_ind + length(x{k});
            end
        end
    end
    

  	switch output_type
        case 'linear'
            
        case 'quadratic'
            % skip the bias and input vector
            x_evolution_org = x_evolution(1 + input_size + 1:end, :);
            %pre-locate memory
            x_evolution_plus = zeros((states_size - 1) * states_size / 2 + 2 * states_size + input_size + 1, size(x_evolution_org,2));

            for i = 1:size(x_evolution_org,2)
                X = nchoosek(x_evolution_org(:,i),2);
                X_ = X(:,1).*X(:,2);
                % include bais, inputs, ESN states, squared ESN states, and producs of
                % all combinations of states: states = {x,y}, then we have {1, u, x, y, x^2, y^2, xy}
                x_evolution_plus(:, i) = [x_evolution(:, i); x_evolution_org(:,i).^2; X_];
            end
            x_evolution = x_evolution_plus;
    end
    
    
    % Train ouput layer by solving a system of linear equations 
    Xinv = pseudoinverse(x_evolution, [],'lsqr');
    Wout = target(initLen + 1:end, :)' * Xinv; 
    
    % Add here the Wiener-Hopf solution
 
end

