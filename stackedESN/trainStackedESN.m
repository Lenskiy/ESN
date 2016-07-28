
function [Wout, states, states_evolution_org] = trainStackedESN(input,...
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

    states_evolution = zeros(1 + input_size + states_size,...
                                                      trainLen - initLen);
    % run the reservoir with the data and collect states

    
    for j = 1:trainLen
        u = input(j, :)';
        cur_state_ind = 1;
        states_evolution(cur_state_ind:input_size + 1,j - initLen) = [1; u];
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
                states{k} = x{k};
                states_evolution(cur_state_ind:cur_state_ind + size(x{k},1) - 1, j - initLen) = states{k};
                cur_state_ind = cur_state_ind + length(x{k});
            end
        end
    end
    
    % skip the bias and input vector
    states_evolution_org = states_evolution(1 + input_size + 1:end, :);
    
    switch output_type
        case 'linear'
            
        case 'quadratic'
            % create additional nodes which are products reservior states
            O = [];
            for i = 1:length(states_evolution) 
                X = nchoosek(states_evolution(:,i),2);
                Y = X(:,1).*X(:,2);
                O = [O Y];
            end
            states_evolution = [states_evolution; O];
    end
    
    
    % Train ouput layer by solving a system of linear equations 
    Xinv = pseudoinverse(states_evolution, [],'lsqr');
    Wout = target(initLen + 1:end, :)' * Xinv; 
    
    % Add here the Wiener-Hopf solution
 
end

