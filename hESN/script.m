%% Mackeyglass training/testing data
mackeyglass_data = mackeyglass(2100);
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(1:mid_point, :)';
train_output = mackeyglass_data(2:mid_point + 1, :)';
test_input = mackeyglass_data(mid_point + 1:end - 1, :)';
test_output = mackeyglass_data(mid_point + 2:end, :)';
%% NARMA10
data = NARMA10series(1600);
testp = 0.5;
split_point = round((1-testp) * length(data)); % take testp as hold out test set
train_input  = data(1, 1:split_point,1);
size(train_input)
train_output = data(2, 2:split_point + 1);
test_input   = data(1, split_point + 1:end - 1);
test_output  = data(2, split_point + 2:end);




% ESN TEST

nNodes = 100;

architecture = struct('inputDim',   size(train_input,1), ...
                      'numNodes',   nNodes, ... 
                      'outputDim',  size(train_output,1));      
parameters  = struct('node_type',      'tanh',... 
                     'radius',      0.8, ...
                     'leakage',     0.2, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand');  
esn = ESN(architecture, parameters);
train = Train();

initL = 100;

train.train(esn, train_input, train_output, initL) 
%Y = esn.generate(test_input(1, :), size(test_output(1,:),2), 1);
Y = esn.predict(test_input(1, :), 1);


figure, hold on; title('system');
plot(test_output(1,:));
plot(Y(1,:));


% SESN TEST
sArchitecture = struct('inputDim',  size(train_input,1), ...
                      'numNodes',   [10; 10], ... 
                      'outputDim',  size(train_output,1));
                   
sParameters(1)  = struct('node_type',      'tanh',... 
                     'radius',      0.8, ...
                     'leakage',     0.2, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand');  
sParameters(2)  = struct('node_type',      'tanh',... 
                     'radius',      0.8, ...
                     'leakage',     0.2, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand'); 
                 
sESN = StackedESN(sArchitecture, sParameters, 'rand');
train = Train();
train.train(sESN, train_input, train_output, 100)
Y = esn.generate(test_input(1, :), size(test_output(1,:),2), 1);
%Y = esn.predict(train_input(1, :), 1);
figure, hold on; title('Mackeyglass system');
plot(test_input(1,:));
plot(Y(1,:));





