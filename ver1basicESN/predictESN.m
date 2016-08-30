
function Y = predictESN(input, states, Win, W, Wout, lr)
    % the states depend on the input and previous states 
    outSize = size(Wout,1);
    Y = zeros(length(input), outSize);
    u = input(1, :)';
    for k = 1:length(input) - 1
        states = (1 - lr) * states + lr * tanh( Win * [1; u] + W * states);
        y = Wout*[1;u;states];
        Y(k,:) = y;
        u = input(k + 1, :)'; % predictive mode
    end
end

