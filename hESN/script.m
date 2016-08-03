%% Mackeyglass training/testing data
mackeyglass_data = mackeyglass(2100);
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(1:mid_point, :)';
train_output = mackeyglass_data(2:mid_point + 1, :)';
test_input = mackeyglass_data(mid_point + 1:end - 1, :)';
test_output = mackeyglass_data(mid_point + 2:end, :)';
% ESN TEST
nNodes = 10;
architecture = struct('inputDim',   size(train_input,1), ...
                      'numNodes',   nNodes, ... 
                      'outputDim',  size(train_output,1));      
parameters  = struct('neuron',      'tanh',... 
                     'radius',      0.8, ...
                     'leakage',     0.2, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand');  
esn = ESN(architecture, parameters);
train = Train();
train.train(esn, train_input, train_output, 100) % do not change the state, or be able to restart the state to the begining
%Y = esn.generate(test_input(1, :), size(test_output(1,:),2), 1);
Y = esn.predict(train_input(1, :), 1);
figure, hold on; title('Mackeyglass system');
plot(test_input(1,:));
plot(Y(1,:));

% HESN TEST
hArchitecture = struct('inputDim',  size(train_input,2), ...
                      'numNodes',   nNodes, ... 
                      'outputDim',  size(train_output,2));  