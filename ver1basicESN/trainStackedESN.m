
function [Wout, states, states_evolution] = trainStackedESN(input,...
                                                        target, Win, W, lr, sigma)
    initLen = 0;
    trainLen = size(input,1);
    input_size = size(input, 2);
    states_size = 0;
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
            %Hypebolic tangent
            x{k} = (1 - lr(k)) * x{k} + lr(k) * tanh( Win{k} * [1; u] + W{k} * x{k} ) + sigma * randn(size(x{k},1), 1);
            %Relu
            %x{k} = (1 - lr(k)) * x{k} + lr(k) * max( Win{k} * [1; u] + W * x{k}, 0);
            %logistic
            %x{k} = (1 - lr(k)) * x{k} + lr(k) * tanh( Win{k} * [1; u] + W * x{k})/2 + 0.5; %


            if k > initLen % don't store a first few input states 
                states{k} = x{k};
                states_evolution(cur_state_ind:cur_state_ind + size(x{k},1) - 1, j - initLen) = states{k};
                cur_state_ind = cur_state_ind + length(x{k});
            end
        end
    end
    
    % Train ouput layer by solving a system of linear equations 
    Xinv = pseudoinverse(states_evolution, [],'lsqr');
    Wout = target(initLen + 1:end, :)' * Xinv; 
    
    % Add here the Wiener-Hopf solution
    
    % skip the bias and input vector
    states_evolution = states_evolution(1 + input_size + 1:end, :); 
end

