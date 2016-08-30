
function [Wout, states, states_evolution] = trainESN(input,...
                                                        target, Win, W, lr, sigma)
    initLen = 0;
    trainLen = size(input,1);
    reservoir_size = size(W,1);
    input_size = size(input, 2);
    % allocated memory for the design (collected states) matrix
    states_evolution = zeros(1 + input_size + reservoir_size,...
                                                      trainLen - initLen);

    % run the reservoir with the data and collect states
    x = zeros(reservoir_size,1);
    for k = 1:trainLen
        u = input(k, :)';
        %Hypebolic tangent
        x = (1 - lr) * x + lr * tanh( Win * [1; u] + W * x ) + sigma * randn(size(x,1), 1);
        %Relu
        %x = (1 - lr) * x + lr * max( Win * [1; u] + W * x, 0);
        %logistic
        %x = (1 - lr) * x + lr * tanh( Win * [1; u] + W * x)/2 + 0.5; %
        
        
        if k > initLen % don't store a first few input states 
            states_evolution(:,k - initLen) = [1; u; x];
        end
    end
    
    % Train ouput layer by solving a system of linear equations 
    Xinv = pseudoinverse(states_evolution, [],'lsqr');
    Wout = target(initLen + 1:end, :)' * Xinv; 
    
    % Add here the Wiener-Hopf solution
    
    % skip the bias and input vector
    states = states_evolution(end - reservoir_size + 1:end, end); 
end

