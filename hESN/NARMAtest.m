%% NARMA training/testing data

data = NARMA10series(1600);
size(data)

testp = 0.1;

split_point = round((1-testp) * length(data)); % take testp as hold out test set

train_input  = data(1, 1:split_point,1);
size(train_input)

train_output = data(2, 2:split_point + 1);
test_input   = data(1, split_point + 1:end - 1);
test_output  = data(2, split_point + 2:end);



% ESN TEST

nNodes = 25;

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
train.train(esn, train_input, train_output, 100) 
%Y = esn.generate(test_input(1, :), size(test_output(1,:),2), 1);
Y = esn.predict(train_input(1, :), 1);
dim(Y)

figure, hold on; title('NARMA10 test');
plot(test_input(1,:));
plot(Y(1,:));





